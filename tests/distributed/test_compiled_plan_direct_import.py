from cornstarch.distributed.parallelization import ParallelizationPlan


def test_existing_parallelization_module_path_exposes_compile():
    assert callable(ParallelizationPlan.compile)
