# tests/test_bench_env.py
#
# The optional ``env`` argument on bench() / batch_bench():
#   env=None            -> auto-generated sets (regression: unchanged)
#   env=dict            -> template mode, pinned names stay pinned
#   env=list[dict]      -> every instance supplied, n = len(env)
# plus fail-fast validation when a program variable is missing and
# the vulkan check_err path (reference run against the user's env).

import random
import unittest

from toyc.bench import (batch_bench, bench, _gen_sets, _expr_names,
                        _resolve_sets, summarize, to_csv)
from toyc.vm import Cpu

_SEED = 7


class EnvNoneRegression(unittest.TestCase):
    """env=None must produce exactly the pre-change rows for a seed."""

    def test_bench_results_match_old_generator(self):
        prog = "a + b * 2"
        names = _expr_names(prog)
        n, repeat = 6, 2
        rows = bench(prog, backend="cpu", n=n, repeat=repeat, seed=_SEED)
        sets = _gen_sets(names, n, _SEED)   # the old/no-env code path
        expected = [Cpu.run(prog, env=s, silent=True) for s in sets]
        got = [r["result"] for r in rows]
        self.assertEqual(got, expected * repeat)
        # row bookkeeping untouched
        self.assertEqual([r["n"] for r in rows], [n] * n * repeat)
        self.assertEqual([r["inst"] for r in rows], list(range(n)) * repeat)
        for r in rows:
            self.assertIsNone(r["check_err"])
            self.assertIn(r["backend"], ("cpu",))

    def test_resolve_sets_none_is_gen_sets(self):
        names = ["a", "b"]
        self.assertEqual(_resolve_sets(names, 4, _SEED, None),
                         _gen_sets(names, 4, _SEED))

    def test_resolve_sets_none_is_bit_for_bit(self):
        # same distribution/order: first-use order, uniform 1.0–10.0
        rng = random.Random(_SEED)
        expected = [{n: rng.uniform(1.0, 10.0) for n in ("a", "b")}
                    for _ in range(3)]
        self.assertListEqual(_resolve_sets(["a", "b"], 3, _SEED, None),
                             expected)


class TemplateDict(unittest.TestCase):
    """env=dict pins its names; the rest are auto-generated."""

    def test_all_pinned_constant(self):
        rows = bench("k", backend="cpu", n=6, repeat=2, env={"k": 5})
        self.assertEqual([r["result"] for r in rows], [5] * 12)

    def test_pinned_kept_free_swept(self):
        prog = "x + k"
        n, seed = 5, 3
        rng = random.Random(seed)
        xs = [rng.uniform(1.0, 10.0) for _ in range(n)]
        rows = bench(prog, backend="cpu", n=n, repeat=1, seed=seed,
                     env={"k": 5})
        self.assertEqual([r["result"] for r in rows],
                         [x + 5 for x in xs])

    def test_batch_template(self):
        rows = batch_bench("k * 2", backend="cpu", n=4, repeat=3,
                           env={"k": 7})
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["result"] for r in rows], [14] * 3)
        self.assertEqual([r["n"] for r in rows], [4] * 3)

    def test_env_given_means_no_random_value_added(self):
        # without env, "k" alone would be uniform 1.0–10.0, never a
        # fixed 42 on every row
        rows = bench("k", backend="cpu", n=5, repeat=1, seed=_SEED)
        self.assertFalse(all(abs(r["result"] - 42.0) < 1e-12
                             for r in rows))
        rows = bench("k", backend="cpu", n=5, repeat=1, seed=_SEED,
                     env={"k": 42})
        self.assertEqual([r["result"] for r in rows], [42] * 5)


class ListOfDicts(unittest.TestCase):
    """env=list[dict]: every instance supplied, n = len(env)."""

    ENV = [{"x": 1.0, "k": 4}, {"x": 2.0, "k": 4}, {"x": 3.0, "k": 4}]

    def test_bench_row_count_and_results(self):
        rows = bench("x + k", backend="cpu", n=10, repeat=2, env=self.ENV)
        self.assertEqual(len(rows), 6)          # 3 instances x 2 repeats
        self.assertEqual({r["n"] for r in rows}, {3})
        per_repeat = [5.0, 6.0, 7.0]
        self.assertEqual([r["result"] for r in rows], per_repeat * 2)
        self.assertEqual([r["inst"] for r in rows], [0, 1, 2] * 2)

    def test_batch_cpu(self):
        rows = batch_bench("x + k", backend="cpu", n=5, repeat=3,
                           env=self.ENV)
        self.assertEqual(len(rows), 3)          # one row per repeat
        self.assertEqual([r["result"] for r in rows], [5.0] * 3)
        self.assertEqual({r["n"] for r in rows}, {3})
        for r in rows:
            self.assertAlmostEqual(r["per_eval"],
                                   r["total_time_taken"] / 3)

    def test_summarize_and_csv_schema_unchanged(self):
        rows = bench("x + k", backend="cpu", n=2, repeat=1, env=self.ENV)
        summary = summarize(rows)
        self.assertEqual(summary[0]["n"], 3)
        self.assertEqual(summary[0]["runs"], 3)
        # to_csv must keep the same fieldnames
        import csv, os, tempfile
        path = os.path.join(tempfile.mkdtemp(), "timings.csv")
        try:
            to_csv(rows, path)
            with open(path, newline="") as f:
                header = next(csv.reader(f))
            self.assertEqual(header[0], "backend")
            self.assertEqual(header[:9],
                             ["backend", "program", "mode", "n",
                              "repeat", "inst", "result",
                              "total_time_taken", "per_eval"])
        finally:
            os.remove(path)
            os.rmdir(os.path.dirname(path))


class FailFastValidation(unittest.TestCase):
    """Missing program variables must raise before any timing."""

    def test_bench_missing_variable(self):
        with self.assertRaises(ValueError) as cm:
            bench("x + k", backend="cpu", n=3, repeat=1,
                  env=[{"x": 1.0}, {"x": 2.0}])   # k missing everywhere
        self.assertIn("k", str(cm.exception))

    def test_bench_missing_variable_one_dict(self):
        with self.assertRaises(ValueError):
            bench("x + k", backend="cpu", n=3, repeat=1,
                  env=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0, "k": 4}])

    def test_batch_missing_variable(self):
        with self.assertRaises(ValueError) as cm:
            batch_bench("x / q", backend="cpu", n=3, repeat=1,
                        env=[{"x": 1.0}] * 3)
        self.assertIn("q", str(cm.exception))

    def test_template_pins_cover_program(self):
        # template mode: missing names are filled, so no error — but a
        # missing backend name never reaches the backends either way
        rows = bench("x + k", backend="cpu", n=3, repeat=1, seed=_SEED,
                     env={"k": 2})
        self.assertEqual(len(rows), 3)

    def test_empty_list(self):
        with self.assertRaises(ValueError):
            bench("x", backend="cpu", n=3, repeat=1, env=[])

    def test_bad_types(self):
        for bad in ("x", 7, [("x", 1.0)]):
            with self.assertRaises((TypeError, ValueError)):
                bench("x", backend="cpu", n=3, repeat=1, env=bad)
        # a template dict that pins nothing is just "no pins" — free
        # variables get auto-generated, like env=None
        rows = bench("k", backend="cpu", n=2, repeat=1, seed=_SEED,
                     env={"unused": 1})
        self.assertEqual(len(rows), 2)


def _vulkan_ready():
    try:
        from toyc.vm import GpuVulkan
        GpuVulkan.startup()
        return GpuVulkan
    except Exception:
        return None


@unittest.skipUnless(_vulkan_ready() is not None,
                     "no usable Vulkan driver in this environment")
class VulkanListMode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.GpuVulkan = _vulkan_ready()

    def test_bench_list(self):
        env = [{"x": 1.0, "k": 4}, {"x": 2.0, "k": 4}, {"x": 3.0, "k": 4}]
        rows = bench("x * k", backend="vulkan", n=8, repeat=2, env=env)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["n"] for r in rows}, {3})
        per_repeat = [4.0, 8.0, 12.0]
        self.assertEqual([r["result"] for r in rows], per_repeat * 2)

    def test_batch_list_check_err_uses_user_env(self):
        env = [{"x": 1.0, "k": 4}, {"x": 2.0, "k": 4}, {"x": 3.0, "k": 4}]
        rows = batch_bench("x * k", backend="vulkan", n=3, repeat=2,
                           env=env)
        for r in rows:
            if r.get("batch_index") == 0:
                # reference run is against sets[0] = {"x": 1.0, "k": 4}
                ref = Cpu.run("x * k", env=env[0], silent=True)
                self.assertEqual(r["result"], float(env[0]["x"]
                                                    * env[0]["k"]))
                self.assertIsNotNone(r["check_err"])
                self.assertAlmostEqual(r["check_err"], abs(ref - ref),
                                       places=6)
        self.assertTrue(all(row["chunk_n"] >= 1 for row in rows))

    def test_batch_template(self):
        rows = batch_bench("k", backend="vulkan", n=4, repeat=1,
                           env={"k": 5})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"], 5.0)


if __name__ == "__main__":
    unittest.main()
