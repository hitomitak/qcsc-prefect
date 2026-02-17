# Workflow for observability demo on Miyabi


from src import riken_sqd_de


if __name__ == "__main__":
    riken_sqd_de.serve(
        description="SQD with LUCJ parameter optimization with differential evoluation.",
    )
