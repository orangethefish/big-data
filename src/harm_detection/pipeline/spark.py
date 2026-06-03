from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession

from harm_detection.config import ROOT_DIR


def _local_java_home() -> Path | None:
    bundled_root = ROOT_DIR / "tools" / "jdk21"
    if not bundled_root.exists():
        return None
    for candidate in bundled_root.iterdir():
        java_windows = candidate / "bin" / "java.exe"
        java_unix = candidate / "bin" / "java"
        if os.name == "nt" and candidate.is_dir() and java_windows.exists():
            return candidate
        if os.name != "nt" and candidate.is_dir() and java_unix.exists():
            return candidate
    return None


def get_spark(app_name: str = "generalizable-harmful-video-detection") -> SparkSession:
    warehouse_dir = (ROOT_DIR / ".spark-warehouse").as_posix()
    java_home = _local_java_home()
    python_executable = Path(sys.executable)
    if not python_executable.exists():
        resolved = shutil.which("python") or shutil.which("py")
        if resolved:
            python_executable = Path(resolved)
    if java_home:
        os.environ["JAVA_HOME"] = str(java_home)
        os.environ["PATH"] = f"{java_home / 'bin'};{os.environ.get('PATH', '')}"
    os.environ["SPARK_HOME"] = str(Path(pyspark.__file__).resolve().parent)
    os.environ["PYSPARK_PYTHON"] = str(python_executable)
    os.environ["PYSPARK_DRIVER_PYTHON"] = str(python_executable)
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
