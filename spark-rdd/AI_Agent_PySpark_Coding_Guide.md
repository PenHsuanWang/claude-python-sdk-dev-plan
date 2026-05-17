# AI Agent PySpark Coding Guide
## A Comprehensive Reference for AI Agents Writing Production-Grade PySpark Code

> **Purpose**: This guide defines the complete reasoning framework, decision trees, code templates, and validation checklists that an AI agent MUST follow when generating PySpark code. Every section is mandatory unless explicitly noted as optional.

---

## Table of Contents

1. [Agent Reasoning Framework — 5-Phase Checklist](#1-agent-reasoning-framework--5-phase-checklist)
2. [Phase 1 — Data Structure Intelligence](#2-phase-1--data-structure-intelligence)
3. [Phase 2 — Data Size Intelligence](#3-phase-2--data-size-intelligence)
4. [Phase 3 — Data Quality Framework](#4-phase-3--data-quality-framework)
5. [Phase 4 — Performance Decision Engine](#5-phase-4--performance-decision-engine)
6. [Phase 5 — Code Generation Standards](#6-phase-5--code-generation-standards)
7. [Phase 6 — Data Integration Validation](#7-phase-6--data-integration-validation)
8. [Phase 7 — Computation Result Evaluation](#8-phase-7--computation-result-evaluation)
9. [Phase 8 — Error Handling & Recovery](#9-phase-8--error-handling--recovery)
10. [Agent Workflow Template — End-to-End](#10-agent-workflow-template--end-to-end)
11. [Quick-Reference Decision Trees](#11-quick-reference-decision-trees)
12. [Configuration Reference Card](#12-configuration-reference-card)

---

## 1. Agent Reasoning Framework — 5-Phase Checklist

Before writing a single line of transformation code, an AI agent MUST complete the following mental checklist. Skipping any step is a source of silent bugs, OOM failures, or wrong results.

```
┌─────────────────────────────────────────────────────────┐
│         PRE-CODE REASONING CHECKLIST (5 Gates)          │
├─────┬───────────────────────────────────────────────────┤
│  1  │ UNDERSTAND — What is the business question?       │
│     │ What does "correct" look like for the output?     │
├─────┼───────────────────────────────────────────────────┤
│  2  │ STRUCTURE — What are the schemas of all sources?  │
│     │ Are types compatible? Are nested types present?   │
├─────┼───────────────────────────────────────────────────┤
│  3  │ SIZE — How many rows/GB per source?               │
│     │ What is the right partition strategy?             │
├─────┼───────────────────────────────────────────────────┤
│  4  │ QUALITY — What nulls, duplicates, anomalies exist │
│     │ in the data? Are they acceptable or must be fixed?│
├─────┼───────────────────────────────────────────────────┤
│  5  │ OUTPUT — What is the expected output schema,      │
│     │ row count range, and business sanity check?       │
└─────┴───────────────────────────────────────────────────┘
```

### 1.1 Mandatory Questions Before Writing Code

| Category | Question | Why It Matters |
|---|---|---|
| Business | What is the grain of the output? (1 row per customer? per event?) | Determines dedup strategy and GROUP BY keys |
| Business | Are NULLs acceptable in the output? | Drives null-handling logic |
| Business | Is the result additive or non-additive? | Determines if SUM across partitions is safe |
| Structure | What is the join key? Is it guaranteed unique on both sides? | Determines if result will fan-out unexpectedly |
| Structure | Are there struct/array/map columns to explode? | Changes row count post-explode |
| Size | Are any DataFrames small enough to broadcast? | Critical for join strategy |
| Size | How large is the expected output? | Determines output partition count |
| Quality | Is the source data fresh? When was it last updated? | Prevents stale result computation |
| Quality | Are there known data quality issues in the source? | Prevents garbage-in-garbage-out |

---

## 2. Phase 1 — Data Structure Intelligence

### 2.1 Schema Inspection Routine

Every time an agent receives a new DataFrame, it MUST run the following inspection block BEFORE any transformation:

```python
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType, ArrayType, MapType, StringType, IntegerType,
    LongType, DoubleType, FloatType, DecimalType, BooleanType,
    DateType, TimestampType, NullType
)

def inspect_schema(df: DataFrame, name: str = "df") -> dict:
    """
    Complete schema inspection. Run this on EVERY new DataFrame.
    Returns a dict of findings for downstream decision-making.
    """
    schema = df.schema
    findings = {
        "name": name,
        "column_count": len(schema.fields),
        "nullable_columns": [],
        "non_nullable_columns": [],
        "nested_columns": [],         # struct, array, map
        "numeric_columns": [],
        "string_columns": [],
        "temporal_columns": [],
        "boolean_columns": [],
        "ambiguous_types": [],         # StringType that might be numeric/date
        "potential_id_columns": [],    # likely join keys (ends in _id, _key, _cd)
    }

    NUMERIC_TYPES = (IntegerType, LongType, DoubleType, FloatType, DecimalType)
    TEMPORAL_TYPES = (DateType, TimestampType)
    NESTED_TYPES   = (StructType, ArrayType, MapType)

    for field in schema.fields:
        col_name = field.name
        col_type = field.dataType

        # Nullability
        if field.nullable:
            findings["nullable_columns"].append(col_name)
        else:
            findings["non_nullable_columns"].append(col_name)

        # Type categorization
        if isinstance(col_type, NESTED_TYPES):
            findings["nested_columns"].append(
                {"name": col_name, "type": type(col_type).__name__, "detail": str(col_type)}
            )
        elif isinstance(col_type, NUMERIC_TYPES):
            findings["numeric_columns"].append(col_name)
        elif isinstance(col_type, StringType):
            findings["string_columns"].append(col_name)
            # Flag string columns that look like IDs or categories
            if any(suffix in col_name.lower() for suffix in ["_id", "_key", "_cd", "_code", "_num"]):
                findings["ambiguous_types"].append(col_name)
        elif isinstance(col_type, TEMPORAL_TYPES):
            findings["temporal_columns"].append(col_name)
        elif isinstance(col_type, BooleanType):
            findings["boolean_columns"].append(col_name)

        # Potential join key detection
        if any(pattern in col_name.lower() for pattern in ["_id", "_key", "_cd", "id_", "key_"]):
            findings["potential_id_columns"].append(col_name)

    return findings


# Usage
findings = inspect_schema(df, name="orders")
print(f"Columns: {findings['column_count']}")
print(f"Nullable: {findings['nullable_columns']}")
print(f"Nested:   {findings['nested_columns']}")
print(f"Ambiguous types: {findings['ambiguous_types']}")
```

### 2.2 Type Compatibility Matrix

Before joining two DataFrames on a key column, the agent MUST verify type compatibility:

```python
def check_join_key_compatibility(
    df_left: DataFrame,
    df_right: DataFrame,
    left_key: str,
    right_key: str
) -> dict:
    """
    Verify join key type compatibility. Spark will silently produce empty
    results if you join LongType to StringType — no error, no warning.
    """
    left_type  = dict(df_left.dtypes)[left_key]
    right_type = dict(df_right.dtypes)[right_key]

    result = {
        "left_key":   left_key,
        "right_type": right_key,
        "left_dtype":  left_type,
        "right_dtype": right_type,
        "compatible": left_type == right_type,
        "recommendation": None
    }

    if left_type != right_type:
        # Common mismatch patterns and fixes
        numeric_types = {"int", "bigint", "long", "double", "float", "decimal"}
        left_is_num   = any(t in left_type  for t in numeric_types)
        right_is_num  = any(t in right_type for t in numeric_types)

        if left_is_num and right_type == "string":
            result["recommendation"] = f"Cast left key: df_left.withColumn('{left_key}', col('{left_key}').cast('string'))"
        elif left_type == "string" and right_is_num:
            result["recommendation"] = f"Cast right key: df_right.withColumn('{right_key}', col('{right_key}').cast('string'))"
        elif left_type == "date" and right_type == "timestamp":
            result["recommendation"] = f"Cast left: df_left.withColumn('{left_key}', col('{left_key}').cast('timestamp'))"
        else:
            result["recommendation"] = f"Manual review needed: {left_type} vs {right_type}"

        print(f"[WARNING] Join key type mismatch: {left_key}({left_type}) vs {right_key}({right_type})")
        print(f"  Fix: {result['recommendation']}")

    return result
```

### 2.3 Nested Type Handling Patterns

```python
from pyspark.sql import functions as F

# ── STRUCT columns ────────────────────────────────────────────────
# Access nested field (no explode needed, no row count change)
df_flat = df.select(
    "order_id",
    F.col("address.city").alias("city"),
    F.col("address.zip").alias("zip")
)

# ── ARRAY columns ────────────────────────────────────────────────
# explode: 1 row per array element (row count INCREASES)
df_exploded = df.select(
    "order_id",
    F.explode("items").alias("item")        # each element becomes a row
)
# explode_outer: preserves rows with empty/null arrays
df_exploded_outer = df.select(
    "order_id",
    F.explode_outer("items").alias("item")
)
# posexplode: adds positional index
df_with_pos = df.select(
    "order_id",
    F.posexplode("items").alias("pos", "item")
)

# ── MAP columns ───────────────────────────────────────────────────
# Access specific key
df_map_val = df.select(
    "user_id",
    F.col("properties")["utm_source"].alias("utm_source")
)
# Explode map to key-value rows
df_map_exploded = df.select(
    "user_id",
    F.explode("properties").alias("prop_key", "prop_value")
)

# ── CRITICAL: Track row count change after explode ───────────────
before_count = df.count()
df_exploded = df.select("order_id", F.explode("items").alias("item"))
after_count = df_exploded.count()
print(f"Explode multiplied rows: {before_count} → {after_count} ({after_count/before_count:.1f}x)")
```

### 2.4 Schema Drift Detection

When reading from a source that may change over time:

```python
from pyspark.sql.types import StructType
import json

def detect_schema_drift(
    current_schema: StructType,
    expected_schema: StructType,
    mode: str = "strict"  # "strict" | "additive" | "report_only"
) -> dict:
    """
    Compare current schema against expected baseline.
    Use this at pipeline start to catch upstream schema changes before
    they corrupt downstream tables.
    """
    current_fields  = {f.name: f for f in current_schema.fields}
    expected_fields = {f.name: f for f in expected_schema.fields}

    added_columns   = set(current_fields.keys())  - set(expected_fields.keys())
    removed_columns = set(expected_fields.keys()) - set(current_fields.keys())
    type_changed    = {
        name
        for name in current_fields.keys() & expected_fields.keys()
        if current_fields[name].dataType != expected_fields[name].dataType
    }
    nullable_changed = {
        name
        for name in current_fields.keys() & expected_fields.keys()
        if current_fields[name].nullable != expected_fields[name].nullable
    }

    drift = {
        "added_columns":    list(added_columns),
        "removed_columns":  list(removed_columns),
        "type_changed":     list(type_changed),
        "nullable_changed": list(nullable_changed),
        "has_drift":        bool(added_columns | removed_columns | type_changed | nullable_changed)
    }

    if drift["has_drift"]:
        if mode == "strict" and (removed_columns or type_changed):
            raise ValueError(
                f"[SCHEMA DRIFT — FATAL]\n"
                f"  Removed columns: {removed_columns}\n"
                f"  Type changes:    {type_changed}\n"
                f"  Review upstream source before proceeding."
            )
        elif mode == "additive" and (removed_columns or type_changed):
            raise ValueError(f"[SCHEMA DRIFT] Breaking changes: removed={removed_columns}, type_changed={type_changed}")
        else:
            for col in added_columns:
                print(f"[SCHEMA DRIFT] New column added: {col}")
            for col in removed_columns:
                print(f"[SCHEMA DRIFT — WARNING] Column removed: {col}")
            for col in type_changed:
                old_t = str(expected_fields[col].dataType)
                new_t = str(current_fields[col].dataType)
                print(f"[SCHEMA DRIFT — WARNING] Type changed: {col}: {old_t} → {new_t}")

    return drift


# Define expected schema (store in version control / catalog)
expected = StructType.fromJson(json.loads("""{...}"""))
current  = spark.read.parquet("/data/orders").schema
drift    = detect_schema_drift(current, expected, mode="strict")
```

### 2.5 Explicit Schema Generation Template

The agent MUST NEVER rely on schema inference for production DataFrames read from CSV or JSON. Always define schema explicitly:

```python
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, DoubleType, BooleanType,
    DateType, TimestampType, DecimalType, IntegerType
)

# ── TEMPLATE: Define schema before reading ───────────────────────
orders_schema = StructType([
    StructField("order_id",       LongType(),         nullable=False),
    StructField("customer_id",    LongType(),         nullable=False),
    StructField("order_date",     DateType(),         nullable=False),
    StructField("order_ts",       TimestampType(),    nullable=True),
    StructField("total_amount",   DecimalType(18, 2), nullable=True),
    StructField("currency_cd",    StringType(),       nullable=True),
    StructField("status",         StringType(),       nullable=True),
    StructField("is_cancelled",   BooleanType(),      nullable=True),
    StructField("item_count",     IntegerType(),      nullable=True),
])

df = spark.read.schema(orders_schema).csv("/data/orders/", header=True)

# ── Why explicit schema matters ───────────────────────────────────
# 1. CSV inference reads entire file on driver (OOM risk on large files)
# 2. All-string inference misses type errors silently
# 3. Schema inference disables predicate pushdown for some formats
# 4. Inference is non-deterministic when source has NULLs in early rows
```

---

## 3. Phase 2 — Data Size Intelligence

### 3.1 Row Count & Size Estimation

```python
def estimate_dataframe_size(df: DataFrame, sample_fraction: float = 0.01) -> dict:
    """
    Estimate DataFrame size without full scan.
    Use BEFORE making partition / join strategy decisions.
    
    WARNING: .count() triggers a full scan. Use sample-based estimation
    for very large DataFrames where exact count is not needed upfront.
    """
    # Sample-based approach (fast, approximate)
    sample = df.sample(fraction=min(sample_fraction, 1.0), seed=42)
    sample_count = sample.count()
    
    # Estimate total rows
    estimated_total = int(sample_count / sample_fraction)

    # Estimate row size in bytes via JSON serialization of sample rows
    sample_rows = sample.limit(100).toJSON().collect()
    if sample_rows:
        avg_row_bytes = sum(len(r.encode("utf-8")) for r in sample_rows) / len(sample_rows)
    else:
        avg_row_bytes = 200  # fallback estimate

    estimated_bytes = estimated_total * avg_row_bytes
    estimated_mb    = estimated_bytes / (1024 ** 2)
    estimated_gb    = estimated_mb / 1024

    result = {
        "estimated_row_count": estimated_total,
        "avg_row_bytes":       round(avg_row_bytes, 1),
        "estimated_mb":        round(estimated_mb, 1),
        "estimated_gb":        round(estimated_gb, 3),
    }

    print(f"[SIZE ESTIMATE] ~{estimated_total:,} rows | ~{estimated_mb:.0f} MB | ~{estimated_gb:.2f} GB")
    return result


# Alternative: use Spark catalog stats (accurate after ANALYZE TABLE)
def get_catalog_size_stats(spark, table_name: str) -> dict:
    """Read stats from Spark catalog (requires ANALYZE TABLE to have been run)."""
    try:
        stats = spark.sql(f"DESCRIBE DETAIL {table_name}").collect()[0]
        return {
            "size_bytes":   stats["sizeInBytes"],
            "size_mb":      stats["sizeInBytes"] / (1024**2),
            "num_files":    stats["numFiles"],
        }
    except Exception:
        return {}
```

### 3.2 Partition Count Decision Tree

```
Data Size → Optimal Partition Count

  Uncompressed data size:
  ├── < 128 MB  → 1–4 partitions (consider coalesce to 1 for small outputs)
  ├── 128 MB – 1 GB  → 8–16 partitions (default 128 MB/partition target)
  ├── 1 GB – 10 GB   → 80–160 partitions
  ├── 10 GB – 100 GB → 400–800 partitions
  └── > 100 GB       → 1,000+ partitions; tune via AQE

  Rule of thumb: target 128–256 MB per partition (uncompressed)
  Formula: partition_count = max(2 * num_cores, ceil(data_size_mb / 200))

  Special cases:
  ├── Writing to Delta Lake  → use repartition(n, "partition_col") for file sizing
  ├── After wide shuffle     → AQE handles auto-coalesce if enabled
  ├── After filter that removes 80%+ rows → coalesce() to reduce partitions
  └── Skewed data            → repartition(n) with salt key (see Phase 4)
```

```python
def recommend_partition_count(
    estimated_mb: float,
    num_executor_cores: int,
    target_mb_per_partition: int = 200
) -> dict:
    """Return recommended partition count and repartition strategy."""
    size_based = max(1, int(estimated_mb / target_mb_per_partition))
    core_based = 2 * num_executor_cores
    recommended = max(size_based, core_based)

    strategy = "repartition" if recommended > 1 else "coalesce"

    print(f"[PARTITIONS] Data: {estimated_mb:.0f} MB | Cores: {num_executor_cores}")
    print(f"  Size-based recommendation: {size_based}")
    print(f"  Core-based recommendation: {core_based}")
    print(f"  → Use {strategy}({recommended})")

    return {"recommended": recommended, "strategy": strategy}
```

### 3.3 Broadcast Threshold Evaluation

```python
def should_broadcast(
    df: DataFrame,
    spark,
    manual_size_mb: float = None
) -> bool:
    """
    Decide whether to broadcast this DataFrame.
    Broadcast is safe when DataFrame fits in executor memory.
    
    Default Spark threshold: 10 MB (spark.sql.autoBroadcastJoinThreshold)
    Practical safe threshold for Databricks: 50–200 MB
    """
    BROADCAST_THRESHOLD_MB = float(
        spark.conf.get("spark.sql.autoBroadcastJoinThreshold", "10485760")
    ) / (1024 ** 2)

    if manual_size_mb is None:
        # Use catalog stats if available
        try:
            size_bytes = df._jdf.queryExecution().analyzed().stats().sizeInBytes()
            manual_size_mb = size_bytes / (1024 ** 2)
        except Exception:
            print("[BROADCAST] Cannot determine size automatically. Use manual_size_mb parameter.")
            return False

    should_bcast = manual_size_mb < BROADCAST_THRESHOLD_MB

    print(f"[BROADCAST] DataFrame size: {manual_size_mb:.1f} MB | Threshold: {BROADCAST_THRESHOLD_MB:.1f} MB")
    print(f"  → {'BROADCAST' if should_bcast else 'SORT-MERGE JOIN'}")

    return should_bcast
```

### 3.4 OOM Risk Assessment

```python
def assess_oom_risk(
    estimated_gb: float,
    executor_memory_gb: float,
    num_partitions: int,
    join_type: str = "none"  # "broadcast", "sort_merge", "none"
) -> dict:
    """
    Assess OOM risk before running heavy operations.
    Returns risk level and recommended mitigations.
    """
    risk = "LOW"
    mitigations = []

    # Memory per partition check
    mem_per_partition_gb = estimated_gb / max(num_partitions, 1)
    usable_executor_mem  = executor_memory_gb * 0.6 * 0.5  # 60% Spark pool, 50% execution

    if mem_per_partition_gb > usable_executor_mem:
        risk = "HIGH"
        needed_partitions = int(estimated_gb / (usable_executor_mem * 0.8)) + 1
        mitigations.append(f"Increase partitions to at least {needed_partitions}")

    # Broadcast OOM check
    if join_type == "broadcast" and estimated_gb > 1.0:
        risk = "HIGH"
        mitigations.append("Do not broadcast DataFrames > 1 GB — use sort-merge join")

    # Cross join detection (catastrophic)
    if join_type == "cross":
        risk = "CRITICAL"
        mitigations.append("Cross join will produce N×M rows — verify this is intentional")

    print(f"[OOM RISK] {risk}")
    for m in mitigations:
        print(f"  MITIGATION: {m}")

    return {"risk": risk, "mitigations": mitigations}
```

---

## 4. Phase 3 — Data Quality Framework

Run these checks on every source DataFrame before using it in transformations. Log results; never silently ignore quality failures.

### 4.1 Completeness Check (Null Rate per Column)

```python
def check_completeness(df: DataFrame, thresholds: dict = None) -> DataFrame:
    """
    Compute null rate for every column.
    
    thresholds: dict of {col_name: max_allowed_null_rate}
    e.g., {"customer_id": 0.0, "email": 0.1}
    
    Returns a summary DataFrame and raises on threshold violations.
    """
    from pyspark.sql import functions as F

    total = df.count()
    if total == 0:
        raise ValueError("[DATA QUALITY] DataFrame is EMPTY — zero rows detected")

    null_exprs = [
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]
    null_counts = df.select(null_exprs).collect()[0].asDict()

    rows = []
    violations = []
    for col_name, null_count in null_counts.items():
        null_rate = null_count / total
        threshold = (thresholds or {}).get(col_name, 1.0)  # default: no threshold
        is_violation = null_rate > threshold
        rows.append((col_name, total, null_count, round(null_rate, 4), threshold, is_violation))
        if is_violation:
            violations.append(f"  {col_name}: null_rate={null_rate:.2%} > threshold={threshold:.2%}")

    summary_schema = StructType([
        StructField("column_name",  StringType(), True),
        StructField("total_rows",   LongType(),   True),
        StructField("null_count",   LongType(),   True),
        StructField("null_rate",    DoubleType(), True),
        StructField("threshold",    DoubleType(), True),
        StructField("is_violation", BooleanType(), True),
    ])
    summary_df = spark.createDataFrame(rows, schema=summary_schema)
    summary_df.show(truncate=False)

    if violations:
        print("[DATA QUALITY — COMPLETENESS VIOLATION]")
        for v in violations:
            print(v)
        # Decide: raise or warn based on pipeline criticality
        # raise ValueError("Completeness threshold violated — see above")

    return summary_df
```

### 4.2 Uniqueness Check (Duplicate Key Detection)

```python
def check_uniqueness(df: DataFrame, key_columns: list, sample_dups: int = 20) -> dict:
    """
    Check for duplicate rows on the specified key columns.
    
    Returns duplicate count and examples.
    Raises if duplicates found and key is expected to be unique.
    """
    from pyspark.sql import functions as F
    from pyspark.sql import Window

    total = df.count()

    # Count rows per key
    dup_df = (
        df.groupBy(key_columns)
          .agg(F.count("*").alias("row_count"))
          .filter(F.col("row_count") > 1)
    )

    dup_count = dup_df.count()
    dup_rate  = dup_count / max(total, 1)

    result = {
        "total_rows":       total,
        "duplicate_keys":   dup_count,
        "duplicate_rate":   round(dup_rate, 4),
        "is_unique":        dup_count == 0,
    }

    if dup_count > 0:
        print(f"[DATA QUALITY — UNIQUENESS] FAIL: {dup_count:,} duplicate keys ({dup_rate:.2%} of {total:,} rows)")
        print(f"  Key columns: {key_columns}")
        print(f"  Sample duplicates:")
        dup_df.orderBy(F.col("row_count").desc()).show(sample_dups, truncate=False)
    else:
        print(f"[DATA QUALITY — UNIQUENESS] PASS: All {total:,} rows have unique keys {key_columns}")

    return result
```

### 4.3 Validity Checks (Range, Enum, Format, Regex)

```python
def check_validity(df: DataFrame, rules: list) -> DataFrame:
    """
    Apply business validity rules to a DataFrame.
    
    rules: list of dicts, each with:
      - "column": str
      - "type": "range" | "enum" | "regex" | "not_negative" | "custom"
      - "params": type-specific parameters
      - "severity": "ERROR" | "WARNING"
    
    Example rules:
    [
        {"column": "total_amount",  "type": "not_negative",  "severity": "ERROR"},
        {"column": "status",        "type": "enum",           "params": {"values": ["PENDING","PAID","CANCELLED"]}, "severity": "ERROR"},
        {"column": "email",         "type": "regex",          "params": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"}, "severity": "WARNING"},
        {"column": "age",           "type": "range",          "params": {"min": 0, "max": 150}, "severity": "ERROR"},
        {"column": "order_date",    "type": "range",          "params": {"min": "2000-01-01", "max": "2099-12-31"}, "severity": "WARNING"},
    ]
    """
    from pyspark.sql import functions as F

    violation_frames = []

    for rule in rules:
        col_name  = rule["column"]
        rule_type = rule["type"]
        severity  = rule.get("severity", "ERROR")
        params    = rule.get("params", {})

        if rule_type == "not_negative":
            invalid = df.filter(F.col(col_name) < 0)
            label   = f"{col_name} < 0"

        elif rule_type == "range":
            min_val = params.get("min")
            max_val = params.get("max")
            cond = F.lit(False)
            if min_val is not None:
                cond = cond | (F.col(col_name) < min_val)
            if max_val is not None:
                cond = cond | (F.col(col_name) > max_val)
            invalid = df.filter(cond)
            label   = f"{col_name} outside [{min_val}, {max_val}]"

        elif rule_type == "enum":
            allowed  = params["values"]
            invalid  = df.filter(~F.col(col_name).isin(allowed) & F.col(col_name).isNotNull())
            label    = f"{col_name} not in {allowed}"

        elif rule_type == "regex":
            pattern = params["pattern"]
            invalid = df.filter(~F.col(col_name).rlike(pattern) & F.col(col_name).isNotNull())
            label   = f"{col_name} fails pattern {pattern}"

        elif rule_type == "custom":
            condition = params["condition"]  # a Column expression
            invalid   = df.filter(~condition)
            label     = params.get("label", f"{col_name} custom rule")

        else:
            continue

        count = invalid.count()
        print(f"[VALIDITY — {severity}] {label}: {count:,} violations")
        if count > 0:
            invalid.show(5, truncate=False)
            violation_frames.append((label, severity, count))

    return violation_frames
```

### 4.4 Consistency Check (Cross-Table Referential Integrity)

```python
def check_referential_integrity(
    df_child: DataFrame,
    df_parent: DataFrame,
    child_key: str,
    parent_key: str,
    child_name: str = "child",
    parent_name: str = "parent"
) -> dict:
    """
    Find rows in df_child where the foreign key does not exist in df_parent.
    E.g., orders without a matching customer.
    """
    from pyspark.sql import functions as F

    # Anti-join: keep child rows with NO match in parent
    orphans = df_child.join(
        df_parent.select(F.col(parent_key).alias("__pk")),
        df_child[child_key] == F.col("__pk"),
        how="left_anti"
    )

    orphan_count = orphans.count()
    total_child  = df_child.count()
    orphan_rate  = orphan_count / max(total_child, 1)

    result = {
        "orphan_count": orphan_count,
        "total_rows":   total_child,
        "orphan_rate":  round(orphan_rate, 4),
        "passes":       orphan_count == 0
    }

    if orphan_count > 0:
        print(f"[INTEGRITY] FAIL: {orphan_count:,} {child_name} rows ({orphan_rate:.2%}) "
              f"have no matching {parent_name}.{parent_key}")
        orphans.select(child_key).distinct().show(10)
    else:
        print(f"[INTEGRITY] PASS: All {child_name}.{child_key} values exist in {parent_name}")

    return result
```

### 4.5 Timeliness Check (Data Freshness)

```python
def check_freshness(
    df: DataFrame,
    ts_column: str,
    max_age_hours: float = 24.0,
    timezone: str = "UTC"
) -> dict:
    """
    Verify the data is fresh enough (max timestamp is recent enough).
    Prevents pipelines from running on stale data and producing stale output.
    """
    from pyspark.sql import functions as F

    max_ts_row = df.agg(F.max(F.col(ts_column)).alias("max_ts")).collect()[0]
    max_ts     = max_ts_row["max_ts"]

    if max_ts is None:
        raise ValueError(f"[FRESHNESS] FAIL: Column {ts_column} has no non-null timestamps")

    from datetime import datetime, timezone as tz, timedelta
    now          = datetime.now(tz.utc)
    max_ts_aware = max_ts.replace(tzinfo=tz.utc) if max_ts.tzinfo is None else max_ts
    age_hours    = (now - max_ts_aware).total_seconds() / 3600

    is_fresh = age_hours <= max_age_hours
    result   = {
        "max_timestamp": str(max_ts),
        "age_hours":     round(age_hours, 2),
        "threshold_hours": max_age_hours,
        "is_fresh":      is_fresh
    }

    status = "PASS" if is_fresh else "FAIL"
    print(f"[FRESHNESS — {status}] Max {ts_column}: {max_ts} | Age: {age_hours:.1f}h | Threshold: {max_age_hours}h")

    return result
```

---

## 5. Phase 4 — Performance Decision Engine

### 5.1 Join Strategy Decision Tree

```
Given:
  df_left  = large DataFrame
  df_right = ?

  ┌─────────────────────────────────────────────────────────┐
  │ Is df_right small?                                      │
  │   < autoBroadcastJoinThreshold (default 10MB,           │
  │     practical Databricks: 50–200MB)                     │
  └──────────────┬──────────────────────────────────────────┘
                 │
         YES ───►│ BROADCAST HASH JOIN
                 │   from pyspark.sql.functions import broadcast
                 │   df = df_left.join(broadcast(df_right), key)
                 │   ✓ No shuffle on either side
                 │   ✗ Driver must fit df_right in memory
                 │
         NO  ───►│ Is the data bucketed/sorted on the join key?
                 │         │
                 │   YES ──► BUCKET JOIN (no shuffle if both sides bucketed equally)
                 │   NO  ──► SORT-MERGE JOIN (default for large-large joins)
                 │             ✓ Handles arbitrarily large data
                 │             ✗ Requires full shuffle on both sides
                 │
                 │ Is one side VERY skewed on the join key?
                 │         │
                 │   YES ──► SKEW JOIN (AQE handles automatically if enabled)
                 │           OR manual SALTING (see Phase 4.3)
```

```python
from pyspark.sql import functions as F, DataFrame

def join_with_strategy(
    df_left: DataFrame,
    df_right: DataFrame,
    left_key: str,
    right_key: str,
    join_type: str = "inner",
    right_size_mb: float = None,
    broadcast_threshold_mb: float = 50.0
) -> DataFrame:
    """
    Smart join wrapper that picks the right strategy based on size.
    """
    # Validate key type compatibility first
    check_join_key_compatibility(df_left, df_right, left_key, right_key)

    condition = df_left[left_key] == df_right[right_key]

    if right_size_mb is not None and right_size_mb < broadcast_threshold_mb:
        print(f"[JOIN] Using BROADCAST JOIN (right side = {right_size_mb:.1f} MB)")
        return df_left.join(F.broadcast(df_right), condition, join_type)
    else:
        print(f"[JOIN] Using SORT-MERGE JOIN")
        return df_left.join(df_right, condition, join_type)
```

### 5.2 Skew Detection and Salting

```python
def detect_skew(df: DataFrame, key_column: str, top_n: int = 20) -> dict:
    """
    Detect data skew on a join/group key.
    A skewed key means a few values carry the majority of rows,
    causing some tasks to run 10–100x longer than others.
    """
    from pyspark.sql import functions as F

    total = df.count()
    dist  = (
        df.groupBy(key_column)
          .agg(F.count("*").alias("count"))
          .orderBy(F.col("count").desc())
    )

    top_rows = dist.limit(top_n).collect()
    top_vals = [(r[key_column], r["count"]) for r in top_rows]

    if not top_vals:
        return {"is_skewed": False}

    top1_count   = top_vals[0][1]
    top1_rate    = top1_count / max(total, 1)
    is_skewed    = top1_rate > 0.1  # 10% of rows in one key = skewed

    print(f"[SKEW DETECTION] Key: {key_column} | Total rows: {total:,}")
    print(f"  Top key: {top_vals[0][0]} → {top1_count:,} rows ({top1_rate:.1%} of total)")
    if is_skewed:
        print(f"  [WARNING] SKEW DETECTED — top value holds {top1_rate:.1%} of rows")

    return {"is_skewed": is_skewed, "top_key_rate": top1_rate, "distribution": top_vals}


def join_with_salting(
    df_large: DataFrame,
    df_small: DataFrame,
    key: str,
    salt_buckets: int = 10,
    join_type: str = "inner"
) -> DataFrame:
    """
    Salt-based skew mitigation for large-large joins where one key dominates.
    
    Strategy:
    1. Add random salt (0..N-1) to large DataFrame key
    2. Replicate small DataFrame N times (one copy per salt bucket)
    3. Join on (key, salt)
    """
    from pyspark.sql import functions as F

    # Add salt to large side
    df_large_salted = df_large.withColumn(
        "__salt", (F.rand() * salt_buckets).cast("int")
    ).withColumn("__salted_key", F.concat(F.col(key).cast("string"), F.lit("_"), F.col("__salt")))

    # Replicate small side
    salt_df = spark.range(salt_buckets).withColumnRenamed("id", "__salt")
    df_small_replicated = (
        df_small
        .crossJoin(salt_df)
        .withColumn("__salted_key", F.concat(F.col(key).cast("string"), F.lit("_"), F.col("__salt")))
    )

    # Join on salted key
    joined = df_large_salted.join(df_small_replicated, on="__salted_key", how=join_type)

    # Drop salt columns
    joined = joined.drop("__salt", "__salted_key")

    return joined
```

### 5.3 Caching Decision Rules

```
CACHE a DataFrame when:
  1. It is used in 2 or more actions (count, show, write, etc.)
  2. It is the result of an expensive transformation (many joins, UDFs)
  3. It is used in iterative algorithms or multiple branches of a DAG

DO NOT CACHE when:
  1. DataFrame is used only once
  2. DataFrame fits in one small partition (< 10 MB) — IO overhead outweighs benefit
  3. DataFrame is read from a fast source (Delta cache / SSD) and used once
  4. Memory is scarce — caching evicts other DataFrames from execution memory

ALWAYS unpersist when done:
  df.unpersist()
```

```python
from pyspark.storagelevel import StorageLevel

def cache_with_validation(df: DataFrame, name: str, level: StorageLevel = StorageLevel.MEMORY_AND_DISK) -> DataFrame:
    """
    Cache a DataFrame with logging. Always call unpersist when done.
    
    Storage levels:
    - MEMORY_ONLY:         Fast access; evicted to recompute if no space (risk of re-scan)
    - MEMORY_AND_DISK:     Spills to disk if memory full (recommended default)
    - DISK_ONLY:           Slowest; use when DataFrame is very large but reused many times
    - MEMORY_AND_DISK_SER: Serialized; less memory but slower to read
    """
    df.persist(level)
    # Materialize the cache now (avoid lazy surprise later)
    count = df.count()
    print(f"[CACHE] '{name}' cached | {count:,} rows | Level: {level}")
    return df

# Pattern: always track cached DataFrames for cleanup
_cached_dfs = []

def managed_cache(df: DataFrame, name: str) -> DataFrame:
    df.persist(StorageLevel.MEMORY_AND_DISK)
    _cached_dfs.append((name, df))
    return df

def release_all_caches():
    for name, df in _cached_dfs:
        df.unpersist()
        print(f"[CACHE] Released: {name}")
    _cached_dfs.clear()
```

### 5.4 Shuffle Minimization Checklist

The agent MUST apply these optimizations before finalizing any plan that involves joins or aggregations:

```
SHUFFLE MINIMIZATION CHECKLIST
─────────────────────────────────────────────────────────────
□ 1. FILTER EARLY
     Apply all filter() calls BEFORE joins, not after.
     Reducing rows reduces shuffle data volume.

□ 2. SELECT ONLY NEEDED COLUMNS BEFORE JOIN
     df_left  = df_left.select("key", "col_a", "col_b")
     df_right = df_right.select("key", "col_c")
     Dropping unused columns reduces shuffle payload.

□ 3. AVOID MULTIPLE SHUFFLES ON SAME KEY
     If you GROUP BY the same key multiple times, combine into one:
     BAD:  df.groupBy("k").agg(sum("a")).join(df.groupBy("k").agg(avg("b")), "k")
     GOOD: df.groupBy("k").agg(sum("a"), avg("b"))

□ 4. USE BROADCAST FOR SMALL LOOKUPS
     Any DataFrame < 50 MB that's joined to a large one should be broadcast.

□ 5. USE BUCKETING FOR REPEATED JOINS ON THE SAME KEY
     If the same two tables are joined repeatedly, bucket them once:
     df.write.bucketBy(200, "customer_id").sortBy("customer_id").saveAsTable("...")
     Future joins on customer_id will skip the shuffle stage.

□ 6. CHECK AQE IS ENABLED
     spark.conf.get("spark.sql.adaptive.enabled") == "true"
     AQE auto-coalesces post-shuffle partitions and converts SMJ → BHJ at runtime.

□ 7. AVOID DISTINCT ON LARGE DATASETS UNLESS NECESSARY
     DISTINCT forces a full shuffle. Use dropDuplicates(["key_cols"]) instead,
     and only on the specific columns needed.

□ 8. AVOID CROSS JOINS
     df.crossJoin(df2) creates N×M rows. Always verify this is intentional.
     If needed for cartesian product, limit df2 first.
─────────────────────────────────────────────────────────────
```

### 5.5 AQE Configuration

```python
# Always verify AQE is enabled before running large jobs
def configure_aqe(spark, enable: bool = True):
    """Configure Adaptive Query Execution settings."""
    spark.conf.set("spark.sql.adaptive.enabled",                          str(enable).lower())
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled",       str(enable).lower())
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled",                 str(enable).lower())
    spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled",       str(enable).lower())

    # Tuning parameters
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes",     "128mb")   # target partition size
    spark.conf.set("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1")
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor",   "5")       # 5x median = skewed
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256mb")

    print(f"[AQE] Adaptive Query Execution: {'ENABLED' if enable else 'DISABLED'}")
```

---

## 6. Phase 5 — Code Generation Standards

### 6.1 Mandatory Rules (Never Violate)

```
RULE 1: ALWAYS USE BUILT-IN FUNCTIONS, NOT PYTHON UDFs
  WHY: Python UDFs serialize every row JVM→Python→JVM = 10–100x slower
  DO:  F.upper(col("name"))         — Catalyst-optimized, JVM native
  DO:  F.regexp_replace(...)        — JVM native
  DO:  @F.pandas_udf(returnType)    — vectorized batch, Pandas Arrow bridge
  DON'T: @udf(returnType)           — per-row Python overhead

RULE 2: ALWAYS DEFINE SCHEMA EXPLICITLY FOR CSV/JSON
  WHY: Schema inference reads whole file; produces all-string types for CSV
  DO:  spark.read.schema(my_schema).csv(path)
  DON'T: spark.read.csv(path, inferSchema=True)

RULE 3: ALWAYS PUSH FILTERS BEFORE JOINS
  WHY: Filters reduce shuffle data; Catalyst may not always push through complex plans
  DO:  df_filtered = df.filter(...); df_filtered.join(other, key)
  DON'T: df.join(other, key).filter(...)

RULE 4: ALWAYS HANDLE NULL EXPLICITLY
  WHY: NULL propagates silently; NULL != NULL in join conditions
  DO:  F.coalesce(col("amount"), F.lit(0))
  DO:  df.filter(col("key").isNotNull())
  DO:  df.na.fill({"amount": 0, "status": "UNKNOWN"})
  DON'T: Assume absence of NULLs without checking

RULE 5: ALWAYS UNPERSIST CACHED DATAFRAMES
  WHY: Leaked cached DataFrames consume storage memory forever in a session
  DO:  df.persist(); ...; df.unpersist()
  DON'T: df.cache() in a loop without cleanup

RULE 6: RUN explain() BEFORE PRODUCTION EXECUTION
  WHY: Catches missing broadcasts, unexpected cross joins, extra shuffles
  DO:  df_result.explain(mode="formatted")
  CHECK: Look for BroadcastHashJoin vs SortMergeJoin
         Look for Filter pushed into scan (FileScan with PushedFilters)
         Look for Exchange nodes (each = a shuffle)

RULE 7: NEVER USE .collect() ON LARGE DATAFRAMES
  WHY: collect() brings ALL data to driver — instant OOM for large DataFrames
  DO:  df.show(20)                  — safe sampling
  DO:  df.limit(100).collect()      — safe with explicit limit
  DO:  df.write.parquet(path)       — distribute write across executors
  DON'T: result = df.collect()      — unless df is known small (< 1 GB)

RULE 8: ALWAYS USE COLUMN EXPRESSIONS, NOT PYTHON ITERATION
  WHY: Python loops over DataFrame rows are 1000x slower than Spark column ops
  DO:  df.withColumn("new_col", col("a") + col("b"))
  DON'T: for row in df.collect(): new_val = row.a + row.b  # NEVER
```

### 6.2 Standard PySpark Code Template

```python
from pyspark.sql import SparkSession, DataFrame, functions as F, Window
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType,
    BooleanType, DateType, TimestampType, DecimalType, IntegerType
)
from pyspark.storagelevel import StorageLevel
import logging

logger = logging.getLogger(__name__)

# ── 1. SparkSession (one per application) ────────────────────────
spark = (
    SparkSession.builder
    .appName("pipeline_name")
    # Only set these if NOT running on Databricks (Databricks pre-configures them)
    # .config("spark.sql.adaptive.enabled", "true")
    # .config("spark.sql.shuffle.partitions", "400")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── 2. Define schemas explicitly ─────────────────────────────────
orders_schema = StructType([
    StructField("order_id",    LongType(),         nullable=False),
    StructField("customer_id", LongType(),         nullable=False),
    StructField("order_date",  DateType(),         nullable=False),
    StructField("amount",      DecimalType(18, 2), nullable=True),
    StructField("status",      StringType(),       nullable=True),
])

customers_schema = StructType([
    StructField("customer_id", LongType(),   nullable=False),
    StructField("country",     StringType(), nullable=True),
    StructField("tier",        StringType(), nullable=True),
])

# ── 3. Read with explicit schema ──────────────────────────────────
df_orders    = spark.read.schema(orders_schema).parquet("/data/orders/")
df_customers = spark.read.schema(customers_schema).parquet("/data/customers/")

# ── 4. Filter early (push predicates before joins) ────────────────
df_orders_active = (
    df_orders
    .filter(F.col("status").isin("PAID", "SHIPPED"))          # push filter early
    .filter(F.col("order_date") >= F.lit("2024-01-01"))       # partition pruning
    .filter(F.col("order_id").isNotNull())                    # null guard
    .select("order_id", "customer_id", "order_date", "amount")  # project early
)

# ── 5. Join with size-appropriate strategy ────────────────────────
# customers is small (~5 MB) → broadcast
df_result = df_orders_active.join(
    F.broadcast(df_customers),
    on="customer_id",
    how="left"
)

# ── 6. Null handling on join result ──────────────────────────────
df_result = (
    df_result
    .withColumn("country", F.coalesce(F.col("country"), F.lit("UNKNOWN")))
    .withColumn("tier",    F.coalesce(F.col("tier"),    F.lit("STANDARD")))
    .withColumn("amount",  F.coalesce(F.col("amount"),  F.lit(0.0)))
)

# ── 7. Transformations ───────────────────────────────────────────
df_enriched = (
    df_result
    .withColumn("amount_usd", F.col("amount") * F.lit(1.0))          # example conversion
    .withColumn("order_year",  F.year(F.col("order_date")))
    .withColumn("order_month", F.month(F.col("order_date")))
    .withColumn("is_high_value", F.when(F.col("amount") > 1000, True).otherwise(False))
)

# ── 8. Cache if used multiple times ──────────────────────────────
df_enriched.persist(StorageLevel.MEMORY_AND_DISK)
enriched_count = df_enriched.count()
logger.info(f"Enriched rows: {enriched_count:,}")

# ── 9. Aggregation ───────────────────────────────────────────────
df_summary = (
    df_enriched
    .groupBy("order_year", "order_month", "country", "tier")
    .agg(
        F.count("order_id").alias("order_count"),
        F.sum("amount").alias("total_amount"),
        F.avg("amount").alias("avg_amount"),
        F.countDistinct("customer_id").alias("unique_customers"),
    )
)

# ── 10. Validate result before writing ───────────────────────────
result_count = df_summary.count()
assert result_count > 0, "Summary result is empty — check source data and join conditions"

# ── 11. Write output ─────────────────────────────────────────────
(
    df_summary
    .repartition(10)
    .write
    .mode("overwrite")
    .partitionBy("order_year", "order_month")
    .parquet("/output/order_summary/")
)
logger.info(f"Written {result_count:,} summary rows")

# ── 12. Cleanup ──────────────────────────────────────────────────
df_enriched.unpersist()
logger.info("Pipeline complete")
```

### 6.3 Window Function Template

```python
from pyspark.sql import Window, functions as F

# ── Ranking within a partition ───────────────────────────────────
window_rank = Window.partitionBy("customer_id").orderBy(F.col("order_date").desc())

df_with_rank = df_orders.withColumn("order_rank", F.row_number().over(window_rank))
df_latest    = df_with_rank.filter(F.col("order_rank") == 1)  # most recent order per customer

# ── Running totals / cumulative sum ──────────────────────────────
window_cumsum = (
    Window.partitionBy("customer_id")
          .orderBy("order_date")
          .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)
df_cumulative = df_orders.withColumn("cumulative_amount", F.sum("amount").over(window_cumsum))

# ── Lag/Lead for sequential analysis ─────────────────────────────
window_seq = Window.partitionBy("customer_id").orderBy("order_date")
df_with_lag = (
    df_orders
    .withColumn("prev_order_date", F.lag("order_date", 1).over(window_seq))
    .withColumn("days_since_last",
        F.datediff(F.col("order_date"), F.col("prev_order_date"))
    )
)

# ── Moving average (last 7 days of rows) ─────────────────────────
window_ma = (
    Window.partitionBy("store_id")
          .orderBy("sale_date")
          .rowsBetween(-6, 0)   # current row + 6 preceding rows
)
df_ma = df_sales.withColumn("7d_avg_sales", F.avg("daily_sales").over(window_ma))
```

---

## 7. Phase 6 — Data Integration Validation

Run these checks AFTER all transformations and BEFORE writing the output. They confirm the pipeline produced the right result, not just a non-empty DataFrame.

### 7.1 Source-to-Target Row Count Reconciliation

```python
def reconcile_row_counts(
    source_counts: dict,
    target_count: int,
    join_type: str = "inner",
    expected_ratio: float = None
) -> dict:
    """
    Validate that the output row count makes sense given the inputs.
    
    source_counts: {"orders": 1000000, "customers": 50000}
    target_count:  the output DataFrame count
    join_type:     "inner" | "left" | "outer" | "agg"
    expected_ratio: if provided, assert target/primary_source ≈ expected_ratio
    """
    primary_source = list(source_counts.values())[0]
    actual_ratio   = target_count / max(primary_source, 1)

    print(f"[RECONCILE — ROW COUNT]")
    for name, cnt in source_counts.items():
        print(f"  Source {name}: {cnt:,} rows")
    print(f"  Target:      {target_count:,} rows")
    print(f"  Ratio (target/primary): {actual_ratio:.3f}")

    if join_type == "inner":
        if target_count > primary_source * 1.05:
            print(f"  [WARNING] Inner join result ({target_count:,}) > primary source ({primary_source:,})")
            print(f"  → Possible FANOUT: join key may not be unique on the right side")
    elif join_type == "left":
        if target_count > primary_source * 1.05:
            print(f"  [WARNING] Left join result LARGER than left source → fanout on right side")
        if target_count < primary_source * 0.95:
            print(f"  [WARNING] Left join result SMALLER than left source → unexpected data loss?")
    elif join_type == "agg":
        if target_count >= primary_source:
            print(f"  [WARNING] Aggregation result >= input rows → check GROUP BY keys")

    if expected_ratio is not None:
        tolerance = 0.05
        if abs(actual_ratio - expected_ratio) > tolerance:
            print(f"  [WARNING] Ratio {actual_ratio:.3f} deviates from expected {expected_ratio:.3f} (tolerance ±{tolerance})")

    return {
        "primary_source_count": primary_source,
        "target_count":         target_count,
        "actual_ratio":         actual_ratio,
    }
```

### 7.2 Aggregate Sum Reconciliation

```python
def reconcile_sums(
    df_source: DataFrame,
    df_target: DataFrame,
    metric_columns: list,
    tolerance_pct: float = 0.001   # 0.1% tolerance for floating point
) -> dict:
    """
    Verify that aggregate sums are preserved through the pipeline.
    E.g., total revenue in source == total revenue in target.
    
    IMPORTANT: Only valid for additive metrics (revenue, count, quantity).
    Not valid for averages, percentages, or distinct counts.
    """
    from pyspark.sql import functions as F

    source_sums = df_source.agg(*[F.sum(c).alias(c) for c in metric_columns]).collect()[0].asDict()
    target_sums = df_target.agg(*[F.sum(c).alias(c) for c in metric_columns]).collect()[0].asDict()

    results = {}
    for col_name in metric_columns:
        s_val = source_sums.get(col_name) or 0.0
        t_val = target_sums.get(col_name) or 0.0
        diff  = abs(t_val - s_val)
        pct   = diff / max(abs(s_val), 1e-10)
        passes = pct <= tolerance_pct

        status = "PASS" if passes else "FAIL"
        print(f"[SUM RECONCILE — {status}] {col_name}: source={s_val:,.2f}, target={t_val:,.2f}, diff={diff:,.4f} ({pct:.4%})")
        results[col_name] = {"source": s_val, "target": t_val, "pct_diff": pct, "passes": passes}

    return results
```

### 7.3 Join Match Rate Evaluation

```python
def evaluate_join_match_rate(
    df_left: DataFrame,
    df_right: DataFrame,
    df_joined: DataFrame,
    left_key: str,
    right_key: str,
    min_acceptable_match_rate: float = 0.95
) -> dict:
    """
    After a join, evaluate what percentage of left rows found a match.
    Low match rate = possible key mismatch or data quality issue.
    """
    left_count   = df_left.count()
    right_count  = df_right.count()
    joined_count = df_joined.count()

    # For a left join, non-matched rows will have NULLs from right side
    # We can detect this by checking for a right-only column being null
    right_cols = [c for c in df_right.columns if c != right_key]
    if right_cols:
        check_col = right_cols[0]
        matched_count    = df_joined.filter(F.col(check_col).isNotNull()).count()
        non_matched      = left_count - matched_count
        match_rate       = matched_count / max(left_count, 1)
    else:
        match_rate   = 1.0
        non_matched  = 0

    passes = match_rate >= min_acceptable_match_rate

    status = "PASS" if passes else "FAIL"
    print(f"[JOIN MATCH RATE — {status}]")
    print(f"  Left:    {left_count:,} rows")
    print(f"  Right:   {right_count:,} rows")
    print(f"  Joined:  {joined_count:,} rows")
    print(f"  Matched: {matched_count:,} rows ({match_rate:.2%})")
    print(f"  Unmatched left rows: {non_matched:,} ({1 - match_rate:.2%})")

    if not passes:
        print(f"  [WARNING] Match rate {match_rate:.2%} below threshold {min_acceptable_match_rate:.2%}")
        print(f"  → Investigate: key type mismatch? upstream data quality issue?")

    return {
        "match_rate": match_rate,
        "unmatched":  non_matched,
        "passes":     passes
    }
```

### 7.4 Schema Mapping Validation

```python
def validate_output_schema(
    df_actual: DataFrame,
    expected_schema: StructType,
    mode: str = "strict"  # "strict" | "subset" | "superset"
) -> bool:
    """
    Verify the output DataFrame has the expected schema.
    
    mode:
      "strict"    — exact match of column names, types, and nullable
      "subset"    — all expected columns present; extra columns allowed
      "superset"  — all actual columns are in expected; no extra columns
    """
    actual_fields   = {f.name: f for f in df_actual.schema.fields}
    expected_fields = {f.name: f for f in expected_schema.fields}
    passes = True

    missing = set(expected_fields.keys()) - set(actual_fields.keys())
    extra   = set(actual_fields.keys())   - set(expected_fields.keys())

    if missing:
        print(f"[SCHEMA VALIDATE] MISSING columns: {missing}")
        if mode in ("strict", "subset"):
            passes = False

    if extra and mode == "strict":
        print(f"[SCHEMA VALIDATE] EXTRA columns (not in expected schema): {extra}")

    # Check type mismatches for common columns
    for col_name in set(actual_fields.keys()) & set(expected_fields.keys()):
        actual_type   = actual_fields[col_name].dataType
        expected_type = expected_fields[col_name].dataType
        if str(actual_type) != str(expected_type):
            print(f"[SCHEMA VALIDATE] TYPE MISMATCH: {col_name}: actual={actual_type}, expected={expected_type}")
            if mode == "strict":
                passes = False

    status = "PASS" if passes else "FAIL"
    print(f"[SCHEMA VALIDATE — {status}] mode={mode}")
    return passes
```

---

## 8. Phase 7 — Computation Result Evaluation

These checks ensure the result not only exists but makes business sense.

### 8.1 Statistical Sanity Checks

```python
def statistical_sanity_check(
    df: DataFrame,
    numeric_columns: list,
    context: str = "result"
) -> dict:
    """
    Run statistical sanity checks on numeric output columns.
    Flags: zero totals, negative-only distributions, extreme outliers,
           implausibly large values.
    """
    from pyspark.sql import functions as F

    agg_exprs = []
    for col_name in numeric_columns:
        agg_exprs += [
            F.count(F.col(col_name)).alias(f"{col_name}__count"),
            F.sum(F.col(col_name)).alias(f"{col_name}__sum"),
            F.avg(F.col(col_name)).alias(f"{col_name}__avg"),
            F.min(F.col(col_name)).alias(f"{col_name}__min"),
            F.max(F.col(col_name)).alias(f"{col_name}__max"),
            F.stddev(F.col(col_name)).alias(f"{col_name}__stddev"),
        ]

    stats = df.agg(*agg_exprs).collect()[0].asDict()

    findings = {}
    for col_name in numeric_columns:
        col_stats = {
            "count":  stats.get(f"{col_name}__count", 0),
            "sum":    stats.get(f"{col_name}__sum"),
            "avg":    stats.get(f"{col_name}__avg"),
            "min":    stats.get(f"{col_name}__min"),
            "max":    stats.get(f"{col_name}__max"),
            "stddev": stats.get(f"{col_name}__stddev"),
        }
        issues = []

        if col_stats["sum"] is None or col_stats["count"] == 0:
            issues.append("ALL VALUES ARE NULL")
        else:
            if col_stats["sum"] == 0:
                issues.append("SUM IS ZERO — check for data filtering issue")
            if col_stats["min"] is not None and col_stats["min"] < 0 and col_stats["max"] > 0:
                pass  # mixed positive/negative is often OK (e.g., adjustments)
            elif col_stats["min"] is not None and col_stats["max"] is not None and col_stats["min"] < 0 and col_stats["max"] <= 0:
                issues.append(f"ALL VALUES NEGATIVE (min={col_stats['min']}, max={col_stats['max']}) — verify business logic")
            if col_stats["avg"] is not None and col_stats["stddev"] is not None and col_stats["avg"] != 0:
                cv = abs(col_stats["stddev"] / col_stats["avg"])
                if cv > 100:
                    issues.append(f"EXTREME VARIANCE: CV={cv:.1f} (avg={col_stats['avg']:.2f}, stddev={col_stats['stddev']:.2f})")

        status = "WARN" if issues else "OK"
        print(f"[SANITY — {status}] {col_name}: sum={col_stats['sum']}, avg={col_stats['avg']:.2f}, min={col_stats['min']}, max={col_stats['max']}")
        for issue in issues:
            print(f"  ⚠ {issue}")

        findings[col_name] = {**col_stats, "issues": issues}

    return findings
```

### 8.2 Business Logic Validation

```python
def validate_business_rules(df: DataFrame, rules: list) -> list:
    """
    Validate business-level assertions on the output.
    
    rules: list of dicts:
      {
        "name":        "Revenue must be non-negative",
        "condition":   F.col("total_revenue") >= 0,   # condition that must be TRUE for all rows
        "severity":    "ERROR" | "WARNING",
        "sample_size": 5
      }
    
    Returns list of failed rules.
    """
    from pyspark.sql import functions as F

    failures = []

    for rule in rules:
        # Find rows that VIOLATE the rule (where condition is False)
        violating = df.filter(~rule["condition"])
        viol_count = violating.count()

        if viol_count > 0:
            print(f"[BUSINESS RULE — {rule['severity']}] FAIL: '{rule['name']}'")
            print(f"  {viol_count:,} rows violate this rule")
            violating.show(rule.get("sample_size", 5), truncate=False)
            failures.append({"rule": rule["name"], "violations": viol_count, "severity": rule["severity"]})
        else:
            print(f"[BUSINESS RULE — PASS] '{rule['name']}'")

    return failures


# Example usage
business_rules = [
    {
        "name":      "Revenue is non-negative",
        "condition": F.col("total_revenue") >= 0,
        "severity":  "ERROR"
    },
    {
        "name":      "Avg order value between $1 and $100,000",
        "condition": F.col("avg_order_value").between(1, 100_000),
        "severity":  "WARNING"
    },
    {
        "name":      "Order count is at least 1",
        "condition": F.col("order_count") >= 1,
        "severity":  "ERROR"
    },
    {
        "name":      "Country code is 2 letters",
        "condition": F.length(F.col("country_cd")) == 2,
        "severity":  "WARNING"
    },
]

failures = validate_business_rules(df_summary, business_rules)
error_failures = [f for f in failures if f["severity"] == "ERROR"]
if error_failures:
    raise ValueError(f"[PIPELINE HALT] {len(error_failures)} ERROR-severity business rules failed")
```

### 8.3 Before/After Comparison (Incremental Loads)

```python
def compare_before_after(
    df_before: DataFrame,
    df_after: DataFrame,
    key_columns: list,
    metric_columns: list
) -> dict:
    """
    Compare a metric DataFrame before and after a transformation/load.
    Useful for:
    - Verifying incremental updates didn't corrupt existing data
    - Checking that a reprocessing job changed only what was expected
    - Regression testing pipeline changes
    """
    from pyspark.sql import functions as F

    before_count = df_before.count()
    after_count  = df_after.count()
    count_delta  = after_count - before_count

    print(f"[BEFORE/AFTER] Rows: {before_count:,} → {after_count:,} (Δ {count_delta:+,})")

    # Compare metric sums
    for col_name in metric_columns:
        before_sum = df_before.agg(F.sum(col_name)).collect()[0][0] or 0.0
        after_sum  = df_after.agg(F.sum(col_name)).collect()[0][0]  or 0.0
        delta      = after_sum - before_sum
        pct_change = delta / max(abs(before_sum), 1e-10) * 100
        print(f"  {col_name}: {before_sum:,.2f} → {after_sum:,.2f} (Δ {delta:+,.2f} / {pct_change:+.2f}%)")
        if abs(pct_change) > 10:
            print(f"    [WARNING] Large change: {pct_change:.1f}% — verify expected")

    return {
        "before_count": before_count,
        "after_count":  after_count,
        "count_delta":  count_delta
    }
```

### 8.4 Anomaly Detection (Distribution-based)

```python
def detect_anomalies_iqr(
    df: DataFrame,
    numeric_column: str,
    iqr_multiplier: float = 3.0
) -> dict:
    """
    Detect outliers using the IQR method.
    Flags values below Q1 - k*IQR or above Q3 + k*IQR.
    
    Use for business-critical metrics to catch data ingestion errors
    before they propagate to dashboards.
    """
    from pyspark.sql import functions as F

    quantiles = df.approxQuantile(numeric_column, [0.25, 0.5, 0.75], relativeError=0.01)
    q1, median, q3 = quantiles
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    outliers = df.filter(
        (F.col(numeric_column) < lower_bound) | (F.col(numeric_column) > upper_bound)
    )
    outlier_count = outliers.count()
    total         = df.count()
    outlier_rate  = outlier_count / max(total, 1)

    print(f"[ANOMALY IQR] {numeric_column}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers: {outlier_count:,} rows ({outlier_rate:.2%})")

    if outlier_count > 0 and outlier_rate > 0.01:
        print(f"  [WARNING] High outlier rate — review for data quality issues")
        outliers.select(numeric_column).orderBy(F.col(numeric_column).desc()).show(10)

    return {
        "q1": q1, "q3": q3, "iqr": iqr,
        "lower_bound": lower_bound, "upper_bound": upper_bound,
        "outlier_count": outlier_count, "outlier_rate": outlier_rate
    }
```

---

## 9. Phase 8 — Error Handling & Recovery

### 9.1 Try/Except Wrappers for Pipeline Stages

```python
import traceback
from datetime import datetime

def run_stage(stage_name: str, func, *args, **kwargs):
    """
    Wrap a pipeline stage with structured error handling and timing.
    
    Logs: stage name, start/end time, duration, success/failure.
    On failure: logs full stack trace, re-raises if critical.
    """
    start_time = datetime.now()
    print(f"[STAGE START] {stage_name} @ {start_time.strftime('%H:%M:%S')}")

    try:
        result = func(*args, **kwargs)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[STAGE OK]    {stage_name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[STAGE FAIL]  {stage_name} failed after {elapsed:.1f}s")
        print(f"  Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise  # Re-raise to halt the pipeline


# Usage
def load_orders():
    return spark.read.schema(orders_schema).parquet("/data/orders/")

df_orders = run_stage("load_orders", load_orders)
```

### 9.2 Checkpoint Strategy for Long Lineages

```python
def checkpoint_df(df: DataFrame, path: str, spark) -> DataFrame:
    """
    Persist a DataFrame as a checkpoint to break long DAG lineage.
    
    Use when:
    - DAG depth > 20 stages (risk of StackOverflow in driver)
    - After expensive operations you don't want to recompute on failure
    - In iterative algorithms (ML, graph processing)
    
    Checkpointing physically saves data to HDFS/S3/DBFS and truncates the lineage.
    Unlike persist(), checkpoint() breaks the logical plan entirely.
    """
    # Configure checkpoint directory (do this at SparkSession creation)
    spark.sparkContext.setCheckpointDir(path)

    df_ckpt = df.checkpoint(eager=True)   # eager=True materializes immediately
    print(f"[CHECKPOINT] DataFrame checkpointed to: {path}")
    return df_ckpt


# Long pipeline with checkpoint every N stages
df_step1 = spark.read.parquet("/raw/events/")
df_step2 = df_step1.filter(...)
df_step3 = df_step2.join(df_lookup, "key")
df_step4 = df_step3.groupBy(...).agg(...)

# Checkpoint here — prevents lineage from growing too long
df_checkpointed = checkpoint_df(df_step4, "/checkpoints/pipeline_name/step4/", spark)

df_step5 = df_checkpointed.join(df_other, "key2")
# ...
```

### 9.3 Partial Failure Logging and Retry Pattern

```python
def process_partitions_with_retry(
    input_paths: list,
    process_func,
    max_retries: int = 3,
    spark = None
) -> tuple:
    """
    Process a list of input paths (e.g., date partitions) with retry logic.
    Returns: (successful_results, failed_paths)
    
    Use for pipelines that process many independent partitions where
    partial failure should not halt the entire job.
    """
    import time

    successful = []
    failed     = []

    for path in input_paths:
        attempt = 0
        success = False

        while attempt < max_retries and not success:
            try:
                result = process_func(path, spark)
                successful.append({"path": path, "result": result})
                success = True
            except Exception as e:
                attempt += 1
                wait_sec = 2 ** attempt  # exponential backoff: 2, 4, 8 seconds
                print(f"[RETRY] {path} — attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    print(f"  Retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    print(f"[FAIL] {path} — all {max_retries} attempts exhausted")
                    failed.append({"path": path, "error": str(e)})

    print(f"[SUMMARY] Processed: {len(successful)}/{len(input_paths)} paths | Failed: {len(failed)}")
    if failed:
        print("[FAILED PATHS]")
        for f in failed:
            print(f"  {f['path']}: {f['error']}")

    return successful, failed
```

### 9.4 DataFrame Existence and Empty Guard

```python
def guard_empty(df: DataFrame, name: str, raise_on_empty: bool = True) -> int:
    """
    Check if a DataFrame is empty. Optionally raise.
    Use this before any operation that would silently succeed on empty input
    and produce a misleading empty output.
    """
    count = df.count()
    if count == 0:
        msg = f"[EMPTY GUARD] DataFrame '{name}' has ZERO rows"
        if raise_on_empty:
            raise ValueError(msg + " — pipeline halted")
        else:
            print(f"[WARNING] {msg}")
    else:
        print(f"[EMPTY GUARD] '{name}' OK: {count:,} rows")
    return count


def guard_path_exists(spark, path: str) -> bool:
    """Check if a file/directory path exists before reading."""
    try:
        files = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem \
            .get(spark.sparkContext._jsc.hadoopConfiguration()) \
            .exists(spark.sparkContext._jvm.org.apache.hadoop.fs.Path(path))
        if not files:
            raise FileNotFoundError(f"[PATH GUARD] Path does not exist: {path}")
        return True
    except Exception as e:
        raise FileNotFoundError(f"[PATH GUARD] Cannot access path {path}: {e}")
```

---

## 10. Agent Workflow Template — End-to-End

This is the complete template an AI agent MUST follow for every PySpark task. Do not skip any section.

```python
# ════════════════════════════════════════════════════════════════
# AI AGENT PYSPARK PIPELINE TEMPLATE
# Fill in every section. Remove no section without documented reason.
# ════════════════════════════════════════════════════════════════

from pyspark.sql import SparkSession, DataFrame, functions as F, Window
from pyspark.sql.types import *
from pyspark.storagelevel import StorageLevel
import logging, traceback
from datetime import datetime

logger = logging.getLogger(__name__)

# ── SECTION 0: CONFIGURATION ─────────────────────────────────────
PIPELINE_NAME        = "your_pipeline_name"
INPUT_PATH_PRIMARY   = "/path/to/primary/source"
INPUT_PATH_LOOKUP    = "/path/to/lookup/source"
OUTPUT_PATH          = "/path/to/output"
PARTITION_COLS       = ["year", "month"]  # output partition columns
BROADCAST_THRESHOLD  = 50 * 1024 * 1024  # 50 MB

# ── SECTION 1: SPARK SESSION ──────────────────────────────────────
spark = SparkSession.builder.appName(PIPELINE_NAME).getOrCreate()
spark.sparkContext.setLogLevel("WARN")
# Verify AQE
assert spark.conf.get("spark.sql.adaptive.enabled", "false") == "true", \
    "AQE not enabled — add spark.sql.adaptive.enabled=true to config"

# ── SECTION 2: SCHEMA DEFINITIONS ────────────────────────────────
primary_schema = StructType([
    # FILL IN ALL FIELDS WITH EXPLICIT TYPES
    StructField("id",    LongType(),   nullable=False),
    StructField("value", DoubleType(), nullable=True),
    # ...
])

lookup_schema = StructType([
    StructField("id",   LongType(),   nullable=False),
    StructField("name", StringType(), nullable=True),
    # ...
])

expected_output_schema = StructType([
    # DEFINE OUTPUT SCHEMA BEFORE WRITING
    StructField("id",    LongType(),   nullable=False),
    StructField("value", DoubleType(), nullable=True),
    StructField("name",  StringType(), nullable=True),
])

# ── SECTION 3: READ & STRUCTURE CHECK ────────────────────────────
guard_path_exists(spark, INPUT_PATH_PRIMARY)
guard_path_exists(spark, INPUT_PATH_LOOKUP)

df_primary = spark.read.schema(primary_schema).parquet(INPUT_PATH_PRIMARY)
df_lookup  = spark.read.schema(lookup_schema).parquet(INPUT_PATH_LOOKUP)

# Structure inspection
findings_primary = inspect_schema(df_primary, "primary")
findings_lookup  = inspect_schema(df_lookup, "lookup")

# Schema drift check (skip if first run)
# detect_schema_drift(df_primary.schema, expected_primary_schema, mode="strict")

# ── SECTION 4: SIZE INTELLIGENCE ─────────────────────────────────
size_primary = estimate_dataframe_size(df_primary)
size_lookup  = estimate_dataframe_size(df_lookup)
oom_risk     = assess_oom_risk(
    estimated_gb       = size_primary["estimated_gb"],
    executor_memory_gb = 8,
    num_partitions     = df_primary.rdd.getNumPartitions()
)

# ── SECTION 5: DATA QUALITY CHECKS ───────────────────────────────
guard_empty(df_primary, "primary")
guard_empty(df_lookup, "lookup")

completeness_results = check_completeness(
    df_primary,
    thresholds={"id": 0.0, "value": 0.1}   # id: no nulls allowed; value: 10% max
)
uniqueness_results = check_uniqueness(df_primary, key_columns=["id"])
# check_freshness(df_primary, ts_column="updated_at", max_age_hours=26.0)

# ── SECTION 6: FILTER EARLY ──────────────────────────────────────
# ALWAYS filter before joins
df_primary_filtered = (
    df_primary
    .filter(F.col("id").isNotNull())
    .filter(F.col("value").isNotNull())
    # ADD BUSINESS FILTERS HERE
    .select("id", "value")  # project only needed columns
)

df_lookup_filtered = (
    df_lookup
    .filter(F.col("id").isNotNull())
    .select("id", "name")
)

# ── SECTION 7: JOIN ───────────────────────────────────────────────
# Type compatibility check
check_join_key_compatibility(df_primary_filtered, df_lookup_filtered, "id", "id")

# Choose join strategy
use_broadcast = size_lookup["estimated_mb"] < 50
if use_broadcast:
    df_joined = df_primary_filtered.join(F.broadcast(df_lookup_filtered), on="id", how="left")
    logger.info("Using BROADCAST JOIN for lookup")
else:
    df_joined = df_primary_filtered.join(df_lookup_filtered, on="id", how="left")
    logger.info("Using SORT-MERGE JOIN")

# ── SECTION 8: NULL HANDLING ──────────────────────────────────────
df_clean = (
    df_joined
    .withColumn("name",  F.coalesce(F.col("name"),  F.lit("UNKNOWN")))
    .withColumn("value", F.coalesce(F.col("value"), F.lit(0.0)))
)

# ── SECTION 9: TRANSFORMATIONS ───────────────────────────────────
df_transformed = (
    df_clean
    # ADD BUSINESS TRANSFORMATIONS HERE
    # Use built-in F.* functions only — no Python UDFs
    .withColumn("value_squared", F.col("value") * F.col("value"))
)

# ── SECTION 10: CACHE (if reused) ────────────────────────────────
df_transformed.persist(StorageLevel.MEMORY_AND_DISK)
transformed_count = df_transformed.count()
logger.info(f"Transformed rows: {transformed_count:,}")

# ── SECTION 11: AGGREGATION (if needed) ──────────────────────────
df_agg = (
    df_transformed
    .groupBy("name")
    .agg(
        F.count("id").alias("count"),
        F.sum("value").alias("total_value"),
    )
)

# ── SECTION 12: INTEGRATION VALIDATION ───────────────────────────
primary_count = df_primary_filtered.count()
reconcile_row_counts(
    source_counts = {"primary": primary_count},
    target_count  = transformed_count,
    join_type     = "left"
)
reconcile_sums(df_primary_filtered, df_clean, metric_columns=["value"])

# ── SECTION 13: RESULT EVALUATION ────────────────────────────────
stat_findings = statistical_sanity_check(df_agg, numeric_columns=["total_value", "count"])

business_rules = [
    {"name": "Count is positive",      "condition": F.col("count") > 0,       "severity": "ERROR"},
    {"name": "Total value non-negative","condition": F.col("total_value") >= 0, "severity": "WARNING"},
]
failures = validate_business_rules(df_agg, business_rules)
error_failures = [f for f in failures if f["severity"] == "ERROR"]
if error_failures:
    raise ValueError(f"Business rule violations: {error_failures}")

# ── SECTION 14: SCHEMA VALIDATION ────────────────────────────────
validate_output_schema(df_agg, expected_output_schema, mode="subset")

# ── SECTION 15: EXPLAIN PLAN REVIEW ──────────────────────────────
# Run this before production; comment out after validation
# df_agg.explain(mode="formatted")
# Check for:
#   - BroadcastHashJoin (good for small right side)
#   - SortMergeJoin (expected for large-large joins)
#   - Exchange (each = shuffle; minimize count)
#   - FileScan with PushedFilters (partition pruning working)

# ── SECTION 16: WRITE OUTPUT ──────────────────────────────────────
output_count = df_agg.count()
guard_empty(df_agg, "output", raise_on_empty=True)

(
    df_agg
    .repartition(max(1, output_count // 100_000))  # ~100K rows per file
    .write
    .mode("overwrite")
    .partitionBy(*PARTITION_COLS) if PARTITION_COLS else df_agg.write.mode("overwrite")
).parquet(OUTPUT_PATH)

logger.info(f"[DONE] Written {output_count:,} rows to {OUTPUT_PATH}")

# ── SECTION 17: CLEANUP ───────────────────────────────────────────
df_transformed.unpersist()
logger.info("[CLEANUP] All caches released")
```

---

## 11. Quick-Reference Decision Trees

### 11.1 Join Strategy

```
Q: What is the smaller DataFrame's size?
  < 10 MB   → F.broadcast() — Spark may do it automatically
  10–200 MB → F.broadcast() explicitly, or raise autoBroadcastJoinThreshold
  > 200 MB  → Sort-Merge Join (no hint needed)
              Is data bucketed on join key? → Bucket Join (no shuffle)
              Is data skewed?               → AQE skew join / manual salting
```

### 11.2 Partition Count

```
Q: How big is the DataFrame?
  < 128 MB   → coalesce to 1–4 partitions
  128 MB–1 GB → 8–20 partitions (let AQE tune)
  1–10 GB    → 100–200 partitions
  10–100 GB  → 400–800 partitions
  > 100 GB   → 1000+ partitions; use repartition(n, "skewed_col") for skewed keys
```

### 11.3 When to Cache

```
Q: Is this DataFrame used in 2+ actions?
  YES + expensive compute → .persist(MEMORY_AND_DISK)
  YES + cheap recompute   → skip cache (I/O overhead ≈ recompute cost)
  NO                      → never cache

Q: Is memory pressure high?
  YES → use DISK_ONLY or skip cache
  NO  → MEMORY_AND_DISK is safe
```

### 11.4 UDF vs Built-in

```
Q: Does pyspark.sql.functions have this function?
  YES → Use F.xxx() — always faster, Catalyst-optimized
  NO  → Can I combine existing F.* functions?
          YES → Do so (even if verbose)
          NO  → Use @pandas_udf (vectorized) NOT @udf (per-row)
```

### 11.5 Data Quality Action

```
Null rate > threshold?
  → Decide: fill with default | drop rows | raise error

Duplicate keys found?
  → Decide: deduplicate (dropDuplicates) | aggregate | raise error

Referential integrity failure?
  → Decide: left join (keep orphans) | inner join (drop orphans) | raise error

Freshness failure?
  → Decide: halt pipeline | log warning and proceed | use last known good
```

---

## 12. Configuration Reference Card

### 12.1 Key Spark Configurations

| Configuration Key | Default | Recommended | Purpose |
|---|---|---|---|
| `spark.sql.adaptive.enabled` | `false` (Spark <3.2) / `true` (≥3.2) | `true` | Enable AQE |
| `spark.sql.adaptive.coalescePartitions.enabled` | `true` | `true` | Auto-coalesce post-shuffle |
| `spark.sql.adaptive.skewJoin.enabled` | `true` | `true` | Auto-handle skewed joins |
| `spark.sql.shuffle.partitions` | `200` | `2–4 × num_cores` | Shuffle output partitions |
| `spark.sql.autoBroadcastJoinThreshold` | `10485760` (10 MB) | `52428800` (50 MB) | Broadcast join threshold |
| `spark.sql.files.maxPartitionBytes` | `134217728` (128 MB) | `134217728` | Max bytes per partition on read |
| `spark.serializer` | `JavaSerializer` | `org.apache.spark.serializer.KryoSerializer` | Faster serialization |
| `spark.sql.execution.arrow.pyspark.enabled` | `false` | `true` | Enables Pandas UDF vectorization |
| `spark.sql.parquet.filterPushdown` | `true` | `true` | Parquet predicate pushdown |
| `spark.sql.parquet.mergeSchema` | `false` | `false` | Avoid slow schema merge on read |
| `spark.dynamicAllocation.enabled` | varies | `true` (Databricks default) | Auto-scale executors |
| `spark.driver.memory` | `1g` | `4–16g` | Driver heap size |
| `spark.executor.memory` | `1g` | `8–32g` | Executor heap per executor |
| `spark.memory.fraction` | `0.6` | `0.6` | Fraction of heap for Spark pool |
| `spark.memory.storageFraction` | `0.5` | `0.5` | Storage / Execution split within Spark pool |

### 12.2 Databricks-Specific Optimizations

```python
# Delta Lake optimizations (Databricks default runtime)
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")   # auto-compact small files
spark.conf.set("spark.databricks.delta.autoCompact.enabled",  "true")   # compact on write
spark.conf.set("spark.databricks.io.cache.enabled",           "true")   # Databricks I/O cache (SSD)

# Photon engine (Databricks): enabled at cluster level, no config needed
# To check if Photon is active:
# spark.conf.get("spark.databricks.photon.enabled", "false")

# Delta table read optimization
# Always read from Delta, not raw Parquet, to leverage:
# - Time travel
# - Z-order clustering
# - Data skipping on min/max stats
df = spark.read.format("delta").load("/delta/table/path")

# OPTIMIZE and ZORDER periodically (run as maintenance job)
# spark.sql("OPTIMIZE delta.`/delta/table/path` ZORDER BY (customer_id, order_date)")
```

### 12.3 Memory Anatomy (Unified Memory Manager)

```
JVM Heap (spark.executor.memory = 8 GB example)
│
├── Reserved Memory:  300 MB (hardcoded, internal Spark use)
│
└── Usable: 7.7 GB
    │
    ├── User Memory (40%): 3.08 GB
    │     └── Python objects, Broadcast variables stored in user code,
    │         data structures outside DataFrames
    │
    └── Spark Pool (60%): 4.62 GB
          │
          ├── Execution Memory: up to 4.62 GB (elastic)
          │     └── Shuffle buffers, sort buffers, hash tables, join buffers
          │
          └── Storage Memory: up to 4.62 GB (elastic)
                └── Cached/persisted DataFrames and RDDs

NOTE: Execution and Storage compete for the Spark Pool.
If Execution needs more, it evicts Storage (cached DFs get spilled/recomputed).
Off-heap memory (spark.memory.offHeap.enabled) is separate from this heap.
```

---

## Appendix: Common Anti-Patterns

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `df.collect()` on large DataFrame | OOM on driver | `df.write.parquet(...)` or `df.show(n)` |
| `@udf` per-row Python function | 10–100x slower than built-ins | Use `F.*` built-ins or `@pandas_udf` |
| Schema inference on CSV | Wrong types, slow | `spark.read.schema(explicit_schema).csv(...)` |
| `df.cache()` without `.unpersist()` | Memory leak | Track cached DFs; always unpersist |
| Filter AFTER join | Extra shuffle data | Filter BEFORE join |
| `groupBy().agg()` without dedup check | Double-counting duplicates | Run `check_uniqueness` first |
| `count()` in a loop | Triggers full scan N times | Compute count once; store result |
| Gzip compressed input files | Non-splittable; 1 partition = OOM | Use Snappy or uncompressed; use Parquet |
| `repartition(1)` on large output | Writes 1 huge file; driver bottleneck | `coalesce(n)` for modest reduction; tune n |
| `.distinct()` on full DataFrame | Full shuffle | `dropDuplicates(["key_cols"])` on specific keys |
| Nested loops over DataFrame rows | Serial Python; N×M complexity | Express as DataFrame join or window function |
| Missing null check on join key | Silent wrong results (NULL != NULL in join) | Always filter `isNotNull()` before join |
| Cross join unintentionally | Explodes rows to N×M | Always verify join condition is present |
| Python `datetime` comparison with Spark timestamps | Type mismatch / silent wrong filter | Use `F.lit("2024-01-01").cast(DateType())` |

---

*End of AI Agent PySpark Coding Guide*

> This guide should be treated as a living document. Update the thresholds section when cluster specifications change and add new validation rules as business logic evolves.
