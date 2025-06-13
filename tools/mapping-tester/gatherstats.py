#! /usr/bin/env python3

import argparse
import csv
import json
import os
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor


def parseArguments(args):
    parser = argparse.ArgumentParser(description="Gathers stats after a run")
    parser.add_argument(
        "-o",
        "--outdir",
        default="cases",
        help="Directory to generate the test suite in.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-f",
        "--file",
        type=argparse.FileType("w"),
        default="stats.csv",
        help="The resulting CSV file containing all stats.",
    )
    return parser.parse_args(args)


def run_checked(args):
    r = subprocess.run(args, text=True, capture_output=True)
    if r.returncode != 0:
        print("Command " + " ".join(map(str, args)))
        print(f"Returncode {r.returncode}")
        print(r.stderr)
    r.check_returncode()


def timingStats(dir: pathlib.Path):
    assert dir.is_dir()
    assert (
        os.system("command -v precice-profiling > /dev/null") == 0
    ), 'Could not find the profiling tool "precice-profiling", which is part of the preCICE installation.'
    event_dir = dir / "precice-profiling"
    json_file = dir / "profiling.json"
    timings_file = dir / "timings.csv"

    if not event_dir.is_dir():
        return {}

    try:
        subprocess.run(
            ["precice-profiling", "merge", "--output", json_file, event_dir],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["precice-profiling", "export", "--output", timings_file, json_file],
            check=True,
            capture_output=True,
        )
        import polars as pl

        df = (
            pl.read_csv(timings_file)
            .filter(pl.col("participant") == "B")
            .select("event", "duration", "rank")
        )
        return {
            "globalTime": df.filter(pl.col("event") == "_GLOBAL")
            .select("duration")
            .max()
            .item(),
            "initializeTime": df.filter(pl.col("event") == "initialize")
            .select("duration")
            .max()
            .item(),
            "computeMappingTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.computeMapping.FromA-MeshToB-Mesh$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "PUMcreateClusteringTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.computeMapping.createClustering.FromA-MeshToB-Mesh$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "PUMqueryVerticesTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.computeMapping.queryVertices$"
                )
            )
            .group_by("rank")
            .agg(pl.col("duration").sum().alias("duration_sum"))
            .select(pl.col("duration_sum").max())
            .item(),
            "PUMrbfSolverTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.computeMapping.rbfSolver$"
                )
            )
            .group_by("rank")
            .agg(pl.col("duration").sum().alias("duration_sum"))
            .select(pl.col("duration_sum").max())
            .item(),
            "PUMcomputeWeightsTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.computeMapping.computeWeights$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolverTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.computeMapping.batchedSolver$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolverInitializationTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.solver.initializeKokkos$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolverqueryVerticesTime": df.filter(
                pl.col("event").str.contains("^initialize/map..*.solver.queryVertices$")
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolverCompute2DOffsetsTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.solver.kernel.compute2DOffsets$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolvercopyMeshesTime": df.filter(
                pl.col("event").str.contains("^initialize/map..*.solver.copyMeshes$")
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolvercomputeWeightsTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.solver.kernel.computeWeights$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolvercomputePolynomialQRTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.solver.kernel.computePolynomialQR$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolverAssembleInputMatricesTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.solver.kernel.assembleInputMatrices$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolverAssembleOutputMatricesTime": df.filter(
                pl.col("event").str.contains(
                    "^initialize/map..*.solver.kernel.assembleOutputMatrices$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolverComputeLUTime": df.filter(
                pl.col("event").str.contains("^initialize/map..*.solver.kernel.lu$")
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolverAllocateDataTime": df.filter(
                pl.col("event").str.contains("^initialize/map..*.solver.allocateData$")
            )
            .select("duration")
            .max()
            .item(),
            "mapDataTime": df.filter(
                pl.col("event").str.contains(
                    "^advance/map..*.mapData.FromA-MeshToB-Mesh$"
                )
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolvercopyFromHostToDeviceTime": df.filter(
                pl.col("event").str.contains("^advance/map..*.solver.copyHostToDevice$")
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolvercopyFromDeviceToHostTime": df.filter(
                pl.col("event").str.contains("^advance/map..*.solver.copyDeviceToHost$")
            )
            .select("duration")
            .max()
            .item(),
            "BatchedSolverSolveTime": df.filter(
                pl.col("event").str.contains(
                    "^advance/map..*.solver.kernel.batchedSolve$"
                )
            )
            .select("duration")
            .max()
            .item(),
        }
    except:
        return {}


def memoryStats(dir: pathlib.Path):
    assert dir.is_dir()
    stats = {}
    for P in "A", "B":
        memfile = dir / f"memory-{P}.log"
        total = 0
        if memfile.is_file():
            try:
                with open(memfile, "r") as file:
                    total = sum([float(e) / 1.0 for e in file.readlines()])
            except BaseException:
                pass
        stats[f"peakMem{P}"] = total

    return stats


def mappingStats(dir: pathlib.Path):
    statFiles = list(dir.glob("*.stats.json"))
    if not statFiles:
        return {}

    statFile = statFiles[0]
    assert statFile.is_file()
    with open(statFile, "r") as jsonfile:
        return dict(json.load(jsonfile))


def gatherCaseStats(casedir: pathlib.Path):
    assert casedir.is_dir()
    parts = [casedir.name] + [p.name for p in casedir.parents]
    assert len(parts) >= 4
    ranks, meshes, constraint, mapping = parts[:4]
    meshA, meshB = meshes.split("-")
    ranksA, ranksB = ranks.split("-")

    stats = {
        "mapping": mapping,
        "constraint": constraint,
        "mesh A": meshA,
        "mesh B": meshB,
        "ranks A": ranksA,
        "ranks B": ranksB,
    }
    stats.update(timingStats(casedir))
    stats.update(memoryStats(casedir))
    stats.update(mappingStats(casedir))
    return stats


def main(argv):
    args = parseArguments(argv[1:])

    cases = [d.parent for d in args.outdir.rglob("done")]

    if not cases:
        print(f"No cases found in {args.outdir.absolute()}")
        return 1

    allstats = []

    def wrapper(case):
        print(f"Found: {case.relative_to(args.outdir)}")
        return gatherCaseStats(case)

    with ThreadPoolExecutor() as pool:
        for stat in pool.map(wrapper, cases):
            allstats.append(stat)

    fields = {key for s in allstats for key in s.keys()}
    assert fields
    writer = csv.DictWriter(args.file, fieldnames=sorted(fields))
    writer.writeheader()
    writer.writerows(allstats)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
