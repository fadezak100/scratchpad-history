import json

from scratchpad.config import ExperimentConfig, ModelConfig, TaskConfig, TrainConfig
from scratchpad.task import instance_key
from scratchpad.model import build_model
from scratchpad.train import build_optimizer, lr_at, make_train_instances, train


def tiny_config(tmp_path, k):
    return ExperimentConfig(
        task=TaskConfig(n_eval_per_length=8, n_val=8),
        model=ModelConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_heads=2),
        train=TrainConfig(history_k=k, n_train_instances=16, batch_size=8, max_steps=4,
                          eval_every=2, n_probe_per_length=4, device="cpu"),
        results_dir=str(tmp_path),
    )


def test_training_graphs_do_not_depend_on_k(tmp_path):
    # Invariant 2: every history condition trains on the same graphs.
    a = make_train_instances(tiny_config(tmp_path, 1))
    b = make_train_instances(tiny_config(tmp_path, None))
    assert [instance_key(x) for x in a] == [instance_key(x) for x in b]


def test_lr_schedules():
    constant = TrainConfig(lr=3e-4, lr_schedule="constant", max_steps=100)
    assert lr_at(0, constant) == lr_at(99, constant) == 3e-4
    cosine = TrainConfig(lr=3e-4, min_lr=3e-5, lr_schedule="cosine", max_steps=100)
    assert lr_at(0, cosine) == 3e-4
    assert abs(lr_at(50, cosine) - (3e-4 + 3e-5) / 2) < 1e-12
    assert abs(lr_at(100, cosine) - 3e-5) < 1e-12
    assert all(lr_at(i, cosine) > lr_at(i + 1, cosine) for i in range(100))


def test_weight_decay_skips_one_dimensional_parameters(tmp_path):
    cfg = tiny_config(tmp_path, 1)
    model = build_model(cfg.model, 34)
    decayed, exempt = build_optimizer(model, cfg.train).param_groups
    assert decayed["weight_decay"] == cfg.train.weight_decay and exempt["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decayed["params"]) and all(p.dim() < 2 for p in exempt["params"])
    assert len(decayed["params"]) + len(exempt["params"]) == len(list(model.parameters()))


def test_cosine_schedule_and_clipping_run(tmp_path):
    cfg = tiny_config(tmp_path, None)
    cfg.train.lr_schedule, cfg.train.grad_clip = "cosine", 1.0
    assert train(cfg)["steps"] == 4


def test_no_early_stop_while_validation_accuracy_is_low(tmp_path):
    # An untrained model sits at ~0% for a while; that plateau must not end the run.
    cfg = tiny_config(tmp_path, 1)
    cfg.train.max_steps, cfg.train.eval_every, cfg.train.early_stop_patience = 8, 1, 2
    metrics = train(cfg)
    assert metrics["steps"] == 8 and metrics["stopped_early"] is False


def test_train_writes_config_and_metrics(tmp_path):
    metrics = train(tiny_config(tmp_path, 1))
    run_dir = tmp_path / "k1_n16_s0"
    assert json.loads((run_dir / "config.json").read_text())["train"]["history_k"] == 1
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved["steps"] == metrics["steps"] == 4
    assert set(saved["eval"]) == {"3", "4", "5", "6"}
    assert saved["examples_seen"] == 4 * 8
    assert len(saved["history"]) == 2
    assert saved["stopped_early"] is False
    assert (run_dir / "model.pt").exists()


def test_fresh_data_run_trains_and_records_the_setting(tmp_path):
    cfg = tiny_config(tmp_path, 1)
    cfg.train.fresh_data = True
    metrics = train(cfg)
    assert metrics["steps"] == 4 and metrics["examples_seen"] == 4 * 8
    saved = json.loads((tmp_path / "k1_n16_s0" / "config.json").read_text())
    assert saved["train"]["fresh_data"] is True


def test_cli_flags_for_task_size_and_fresh_data():
    from scratchpad.train import parse_args

    cfg = parse_args(["--k", "1", "--fresh", "--n-nodes", "16", "--train-steps", "6",
                      "--test-steps", "6,8,10,12", "--eval-every", "500", "--results-dir", "x"])
    assert (cfg.task.n_nodes, cfg.task.train_steps, cfg.task.test_steps) == (16, 6, (6, 8, 10, 12))
    assert cfg.train.fresh_data and cfg.train.eval_every == 500 and cfg.results_dir == "x"
    assert parse_args([]).train.fresh_data is False and parse_args([]).task.n_nodes == 8


def test_first_reaching_returns_the_first_evaluation_at_or_above_the_threshold():
    from scratchpad.utils import first_reaching

    history = [
        {"step": 500, "val_trace_accuracy": 0.0, "examples_seen": 10},
        {"step": 1000, "val_trace_accuracy": 0.95, "examples_seen": 20},
        {"step": 1500, "val_trace_accuracy": 0.80, "examples_seen": 30},
        {"step": 2000, "val_trace_accuracy": 1.0, "examples_seen": 40},
    ]
    assert first_reaching(history, 0.95)["step"] == 1000
    assert first_reaching(history, 0.99)["step"] == 2000
    assert first_reaching(history, 1.01) is None
    assert first_reaching([], 0.5) is None


def test_model_shape_flags_override_the_defaults():
    from scratchpad.model import build_model, count_parameters
    from scratchpad.tokens import VOCAB
    from scratchpad.train import parse_args

    cfg = parse_args(["--n-layers", "6", "--n-heads", "6", "--hidden-size", "384"])
    assert (cfg.model.num_layers, cfg.model.num_heads) == (6, 6)
    assert (cfg.model.hidden_size, cfg.model.intermediate_size) == (384, 1024)  # feed-forward width unchanged
    assert parse_args(["--ffn-size", "1536"]).model.intermediate_size == 1536
    assert parse_args([]).model.hidden_size == 256  # defaults untouched
    # 6 layers / 6 heads / width 384 with the default feed-forward width is about 10.6M parameters,
    # the size of the model in Abbe et al. (2024).
    assert 10_000_000 < count_parameters(build_model(cfg.model, len(VOCAB))) < 11_000_000


def test_true_positions_change_nothing_for_full_history_and_do_for_a_window():
    # With full history nothing is hidden, so the control must compute exactly the same function.
    import random

    import torch

    from scratchpad.data import ExampleDataset, collate
    from scratchpad.task import make_instance
    from scratchpad.tokens import VOCAB

    torch.manual_seed(0)
    model = build_model(ModelConfig(hidden_size=32, intermediate_size=64, num_layers=2, num_heads=2), len(VOCAB))
    model.eval()
    instances = [make_instance(8, 3, True, random.Random(i)) for i in range(3)]

    def losses(k, true_positions):
        batch = collate(ExampleDataset(instances, k, true_positions).examples)
        with torch.no_grad():
            return model(**batch).loss.item()

    assert abs(losses(None, False) - losses(None, True)) < 1e-6
    assert abs(losses(1, False) - losses(1, True)) > 1e-6


def test_time_limit_stops_training_and_still_saves(tmp_path):
    cfg = ExperimentConfig(results_dir=str(tmp_path))
    cfg.task.n_eval_per_length, cfg.task.n_val, cfg.train.n_probe_per_length = 8, 8, 0
    cfg.model = ModelConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_heads=2)
    cfg.train.n_train_instances, cfg.train.batch_size, cfg.train.device = 64, 16, "cpu"
    cfg.train.max_steps, cfg.train.eval_every = 100000, 100000
    cfg.train.true_positions, cfg.train.history_k = True, 1
    cfg.train.time_limit_minutes = 0.01  # under a second
    metrics = train(cfg)
    assert metrics["stopped_by_time_limit"] and metrics["steps"] < 100000
    assert (tmp_path / "k1tp_n64_s0" / "metrics.json").exists()
