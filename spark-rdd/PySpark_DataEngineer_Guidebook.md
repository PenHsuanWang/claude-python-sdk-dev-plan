# PySpark Data Engineer Guidebook
### Best Practices for High-Performance Data Pipelines on Databricks

> **Sources:** Apache Spark Official Documentation (spark.apache.org/docs/latest), PySpark API Reference, and textbook-level RDD & Architecture fundamentals.

---

## Table of Contents

1. [The Spark Mental Model](#1-the-spark-mental-model)
2. [SparkSession — The Entry Point](#2-sparksession--the-entry-point)
3. [Lazy Evaluation: Transformations vs Actions](#3-lazy-evaluation-transformations-vs-actions)
4. [RDD Fundamentals](#4-rdd-fundamentals)
5. [DataFrame API — Core Operations](#5-dataframe-api--core-operations)
6. [PySpark Built-in Functions (`pyspark.sql.functions`)](#6-pyspark-built-in-functions-pysparkssqlfunctions)
7. [Schema Management & Data Types](#7-schema-management--data-types)
8. [Handling Null Values](#8-handling-null-values)
9. [Join Strategies & Optimization](#9-join-strategies--optimization)
10. [Aggregations & Window Functions](#10-aggregations--window-functions)
11. [Partitioning: Repartition vs Coalesce](#11-partitioning-repartition-vs-coalesce)
12. [Caching & Persistence](#12-caching--persistence)
13. [UDFs vs Built-ins — The Performance Choice](#13-udfs-vs-built-ins--the-performance-choice)
14. [Data I/O Best Practices](#14-data-io-best-practices)
15. [Adaptive Query Execution (AQE)](#15-adaptive-query-execution-aqe)
16. [Shuffle Internals & Optimization](#16-shuffle-internals--optimization)
17. [Memory Management Deep Dive](#17-memory-management-deep-dive)
18. [Data Skew Detection & Remediation](#18-data-skew-detection--remediation)
19. [Shared Variables: Broadcast & Accumulators](#19-shared-variables-broadcast--accumulators)
20. [Spark UI & Monitoring](#20-spark-ui--monitoring)
21. [Databricks-Specific Best Practices](#21-databricks-specific-best-practices)
22. [Key Configuration Reference](#22-key-configuration-reference)
23. [Complete Production Pipeline Example](#23-complete-production-pipeline-example)

---

## 1. The Spark Mental Model

Apache Spark is a **unified, in-memory distributed computing engine**. It does not store data — it processes data distributed across a cluster of machines. Understanding its physical architecture is the prerequisite for writing performant code.

### 1.1 Cluster Topology

```
┌──────────────────────────────────────────────────────────┐
│                     DRIVER PROGRAM                       │
│  ┌─────────────────┐    ┌──────────────────────────────┐ │
│  │  SparkSession   │    │  DAG Scheduler               │ │
│  │  SparkContext   │───▶│  Task Scheduler              │ │
│  │  (Entry Point)  │    │  (Builds & Dispatches Plan)  │ │
│  └─────────────────┘    └──────────────────────────────┘ │
└──────────────────────┬───────────────────────────────────┘
                       │  coordinates via RPC
              ┌────────▼────────┐
              │  Cluster Manager│  (YARN / Kubernetes / Standalone)
              │  (Allocates CPU │
              │   & Memory)     │
              └────────┬────────┘
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ Worker Node │ │ Worker Node │ │ Worker Node │
   │  ┌────────┐ │ │  ┌────────┐ │ │  ┌────────┐ │
   │  │Executor│ │ │  │Executor│ │ │  │Executor│ │
   │  │(JVM)   │ │ │  │(JVM)   │ │ │  │(JVM)   │ │
   │  │Task│T  │ │ │  │Task│T  │ │ │  │Task│T  │ │
   │  │Cache   │ │ │  │Cache   │ │ │  │Cache   │ │
   │  └────────┘ │ │  └────────┘ │ │  └────────┘ │
   └─────────────┘ └─────────────┘ └─────────────┘
```

### 1.2 Key Vocabulary

| Term | Role |
|---|---|
| **Driver Program** | Runs `main()`, holds `SparkSession`, builds the execution plan, dispatches tasks, aggregates results |
| **SparkContext** | The gateway between user code and the Spark cluster; created inside `SparkSession` |
| **Cluster Manager** | External daemon (YARN, Kubernetes, Standalone) that provisions CPU/Memory for Executors |
| **Worker Node** | Physical or virtual server that hosts Executor JVM processes |
| **Executor** | A persistent JVM on a Worker Node that runs Tasks and manages in-memory/disk caching |
| **Job** | One top-level parallel computation spawned by a single **Action** |
| **Stage** | A subset of a Job; stages are separated at shuffle (wide-dependency) boundaries |
| **Task** | The smallest unit of execution — processes exactly **one partition** on one Executor thread |
| **Partition** | The fundamental unit of parallelism — one partition = one Task |

> **Critical Rule:** `collect()` and `toPandas()` pull **all** distributed data back to the single Driver. On a large dataset, this will crash the Driver with an Out-of-Memory error. Only collect small, aggregated results.

---

## 2. SparkSession — The Entry Point

`SparkSession` is the unified entry point to all Spark functionality since Spark 2.0. It internally creates and manages `SparkContext`, `SQLContext`, and `HiveContext`.

### 2.1 Initializing SparkSession

```python
from pyspark.sql import SparkSession

# Standard initialization
spark = (SparkSession.builder
    .appName("MyDataPipeline")
    .config("spark.sql.adaptive.enabled", "true")          # Enable AQE (default true in 3.2+)
    .config("spark.sql.shuffle.partitions", "200")         # Default shuffle partitions
    .config("spark.sql.autoBroadcastJoinThreshold", "10m") # Broadcast tables < 10MB
    .getOrCreate())

# Access underlying SparkContext if needed for RDD operations
sc = spark.sparkContext
```

> **Databricks Note:** On Databricks, `spark` is a pre-configured `SparkSession` available in every notebook cell. Do **not** create a new one — `getOrCreate()` safely returns the existing session.

### 2.2 Runtime Configuration

```python
# Set config at runtime (most settings take effect immediately)
spark.conf.set("spark.sql.shuffle.partitions", "400")

# Read current config
spark.conf.get("spark.sql.shuffle.partitions")

# Stop session (only needed outside Databricks managed environments)
spark.stop()
```

---

## 3. Lazy Evaluation: Transformations vs Actions

This is the **single most important concept** in Spark. All your code is building a plan, not executing it.

### 3.1 How It Works

```
User Code                     Spark Internals
─────────────                 ───────────────
df.filter(...)    ──────▶     [Queue: FILTER into Logical Plan]   ← no data moved
  .join(...)      ──────▶     [Queue: JOIN into Logical Plan]     ← no data moved
  .groupBy(...)   ──────▶     [Queue: GROUPBY into Logical Plan]  ← no data moved
  .count()        ──────▶     [ACTION! → Catalyst Optimizer runs]
                              [Physical Plan generated]
                              [Tasks dispatched to Executors]
                              [RESULT returned to Driver]          ← data processed NOW
```

### 3.2 Transformations (Lazy — Return a New DataFrame)

**Narrow Transformations** *(no network shuffle — fast)*

| Function | Description |
|---|---|
| `df.select(*cols)` | Project specific columns |
| `df.filter(condition)` / `df.where(condition)` | Filter rows (identical methods) |
| `df.withColumn(name, expr)` | Add or replace a column |
| `df.withColumnRenamed(old, new)` | Rename a column |
| `df.drop(*cols)` | Remove columns |
| `df.limit(n)` | Return first n rows (lazy) |
| `df.alias(name)` | Assign alias for use in joins |
| `df.map(f)` / `df.flatMap(f)` | Row-level transformation (RDD-level) |

**Wide Transformations** *(require network shuffle — expensive)*

| Function | Description |
|---|---|
| `df.join(other, on, how)` | Join two DataFrames |
| `df.groupBy(*cols)` | Group for aggregation |
| `df.orderBy(*cols)` / `df.sort(*cols)` | Global sort (forces full shuffle) |
| `df.distinct()` | Remove duplicate rows |
| `df.union(other)` / `df.unionByName(other)` | Combine DataFrames vertically |
| `df.repartition(n)` | Redistribute into n partitions (shuffle) |
| `df.coalesce(n)` | Reduce partitions without full shuffle |
| `df.cube(*cols)` | Multi-dimensional aggregation |
| `df.rollup(*cols)` | Hierarchical aggregation |

### 3.3 Actions (Eager — Trigger Execution)

| Function | Returns | Warning |
|---|---|---|
| `df.show(n, truncate)` | Print to console | For debugging only |
| `df.collect()` | `List[Row]` to Driver | ⚠️ OOM risk on large data |
| `df.take(n)` | `List[Row]` — first n | Safer than `collect()` |
| `df.head(n)` / `df.first()` | Row(s) to Driver | Triggers full job |
| `df.count()` | `int` | Full table scan |
| `df.describe(*cols)` | Stats DataFrame | Full scan |
| `df.write.save(path)` | Write to storage | Full computation |
| `df.toPandas()` | Pandas DataFrame | ⚠️ All data to Driver |

---

## 4. RDD Fundamentals

RDDs are the **foundational abstraction** — every DataFrame compiles down to RDD operations. Understanding RDDs explains *why* DataFrames behave as they do.

### 4.1 Five Internal Properties of an RDD

The Spark scheduler reads these 5 properties to map logic to hardware:

| Property | Purpose |
|---|---|
| **Parent Dependencies** | List of parent RDDs; constructs the lineage DAG edges |
| **Partition Array** | Defines the logical partitions (1 partition = 1 Task on 1 Executor thread) |
| **Compute Function** | The actual logic applied to each partition when a task runs |
| **Partitioner** (optional) | How keys are distributed: `HashPartitioner` or `RangePartitioner` |
| **Preferred Locations** (optional) | Data locality hints — e.g., HDFS DataNode IPs hosting the block |

### 4.2 Lineage and Fault Tolerance

Spark does **not** replicate data for fault tolerance. Instead, it maintains a **lineage DAG** — a directed acyclic graph recording every transformation from the source data. If a partition is lost, Spark traces back through the DAG and **recomputes only that partition**.

```python
# Example: 3-step lineage
rdd1 = sc.textFile("s3://bucket/raw/")        # Parent
rdd2 = rdd1.filter(lambda x: len(x) > 10)     # +1 edge
rdd3 = rdd2.map(lambda x: x.upper())          # +1 edge
# Lineage: rdd3 ← rdd2 ← rdd1
# If an rdd3 partition fails, Spark recomputes from rdd1 only for that partition
```

> **Checkpoint long lineages:** For iterative ML jobs where lineage grows very deep, use `df.checkpoint()` to break the chain and save state to HDFS. This prevents catastrophic recomputation on failure.

```python
# Set checkpoint directory first (required)
sc.setCheckpointDir("dbfs:/checkpoints/pipeline_v1/")
df_cleaned.checkpoint()  # Saves to HDFS, breaks lineage
```

### 4.3 Narrow vs Wide Dependencies

This determines whether a shuffle (network transfer) is needed:

```
NARROW DEPENDENCY                     WIDE DEPENDENCY
(no shuffle)                          (requires shuffle)

Partition 1 ──▶ Partition 1'         Partition 1 ─┐
Partition 2 ──▶ Partition 2'                       ├──▶ Partition A'
Partition 3 ──▶ Partition 3'         Partition 2 ─┘
                                     Partition 3 ─┐
filter(), select(), map()                          ├──▶ Partition B'
withColumn(), drop()                 Partition 4 ─┘

                                     groupBy(), join(),
                                     distinct(), orderBy()
```

> **Stage boundary rule:** Every wide transformation creates a new Stage. `N` wide dependencies in your DAG = `N+1` Stages. Each stage boundary requires writing intermediate data to local disk before the next stage can begin.

### 4.4 When to Use RDDs

Prefer DataFrames/Spark SQL almost always. Use raw RDDs **only** when:
- You need to manipulate **unstructured data** (binary, free text)
- You require **fine-grained control** over partitioning not expressible in SQL
- You are building **custom transformations** that Catalyst can't reason about

```python
# RDD creation
rdd = sc.parallelize([1, 2, 3, 4, 5], numSlices=4)
rdd_from_file = sc.textFile("s3://bucket/logs/*.txt")

# Basic RDD operations
rdd.map(lambda x: x * 2).collect()
rdd.filter(lambda x: x % 2 == 0).take(3)
rdd.reduce(lambda a, b: a + b)
rdd.saveAsTextFile("s3://bucket/output/")
```

---

## 5. DataFrame API — Core Operations

### 5.1 Creating DataFrames

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# From in-memory data (dict list)
df = spark.createDataFrame([
    {"id": 1, "name": "Alice", "salary": 90000.0},
    {"id": 2, "name": "Bob",   "salary": 75000.0},
])

# From an explicit schema (recommended for production)
schema = StructType([
    StructField("id",     IntegerType(), nullable=False),
    StructField("name",   StringType(),  nullable=True),
    StructField("salary", DoubleType(),  nullable=True),
])
df = spark.createDataFrame(data, schema=schema)

# From files (most common in pipelines)
df = spark.read.format("parquet").load("s3://bucket/data/")
df = spark.read.format("delta").load("dbfs:/mnt/delta/transactions/")
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("...")

# From a SQL table
df = spark.table("catalog.schema.table_name")

# From a SQL query
df = spark.sql("SELECT * FROM catalog.schema.table WHERE event_date = '2026-05-17'")
```

### 5.2 Inspecting DataFrames

```python
df.printSchema()          # Print column names, types, nullability
df.schema                 # Returns StructType object
df.columns                # List of column names
df.dtypes                 # List of (name, type) tuples
df.count()                # Number of rows (triggers Action)
df.show(5, truncate=False) # Print first 5 rows with full strings
df.describe("salary").show() # Count, mean, stddev, min, max

# Inspect the execution plan — CRITICAL for performance debugging
df.explain()              # Physical plan
df.explain("extended")    # Logical + Physical plans
df.explain("cost")        # With optimizer cost estimates
df.explain("formatted")   # Human-readable tree format (Spark 3.0+)
```

### 5.3 Selecting and Projecting Columns

```python
import pyspark.sql.functions as F

# Basic column selection
df.select("id", "name", "salary")
df.select(df.id, df.name, df["salary"])
df.select(F.col("id"), F.col("name"))

# Computed columns inline
df.select(
    F.col("name"),
    (F.col("salary") * 1.1).alias("salary_with_raise"),
    F.upper(F.col("name")).alias("name_upper")
)

# Select all + computed
df.select("*", F.round(F.col("salary") / 1000, 2).alias("salary_k"))

# Regex-based column select
df.select(df.colRegex("`salary.*`"))   # selects all columns starting with "salary"

# Drop columns
df.drop("temp_col", "another_col")
```

### 5.4 Filtering Rows

```python
# String expression (SQL-like, most readable)
df.filter("salary > 50000 AND name IS NOT NULL")
df.where("event_date = '2026-05-17'")

# Column expression (preferred — type-safe, composable)
df.filter(F.col("salary") > 50000)
df.filter((F.col("dept") == "Engineering") & (F.col("salary") > 80000))
df.filter(F.col("status").isin("active", "probation"))
df.filter(F.col("name").isNotNull())
df.filter(F.col("code").like("TX%"))          # SQL LIKE
df.filter(F.col("desc").rlike(r"^\d{4}-\d{2}"))  # Regex
df.filter(F.col("id").between(100, 200))

# Filter nulls
df.filter(F.col("email").isNull())
df.filter(F.col("email").isNotNull())
```

### 5.5 Adding and Transforming Columns

```python
# Add a new column
df.withColumn("salary_eur", F.col("salary") * 0.92)

# Replace an existing column (same name)
df.withColumn("salary", F.round(F.col("salary"), 2))

# Rename
df.withColumnRenamed("salary", "annual_salary")

# Conditional logic with when/otherwise (equivalent to SQL CASE WHEN)
df.withColumn("band",
    F.when(F.col("salary") >= 100000, "Senior")
     .when(F.col("salary") >= 70000,  "Mid")
     .otherwise("Junior")
)

# Type casting
df.withColumn("id_str", F.col("id").cast(StringType()))
df.withColumn("amount", F.col("amount_str").cast("double"))

# Multiple transformations chained (preferred over multiple withColumn calls)
df = (df
    .withColumn("salary_eur", F.round(F.col("salary") * 0.92, 2))
    .withColumn("dept_code",  F.upper(F.col("dept").substr(1, 3)))
    .withColumn("hire_year",  F.year(F.col("hire_date")))
    .filter(F.col("is_active") == True)
)
```

### 5.6 Sorting

```python
df.orderBy("salary")                         # Ascending default
df.orderBy(F.col("salary").desc())           # Descending
df.orderBy(F.col("dept").asc(), F.col("salary").desc())  # Multi-column
df.sort(F.col("salary").desc_nulls_last())   # Control null position
```

> ⚠️ **`orderBy()` is a wide transformation** — it forces a full global shuffle to produce a globally sorted output. Use it only when truly needed, and as **late in the pipeline as possible**. For display purposes use `df.show()` with a preceding `.limit()`.

---

## 6. PySpark Built-in Functions (`pyspark.sql.functions`)

**Always import as:** `import pyspark.sql.functions as F`

The Catalyst Optimizer and Tungsten engine can only **see inside** built-in functions. They execute directly in the JVM — no Python serialization penalty. This is the #1 performance principle.

### 6.1 String Functions

```python
F.upper(col)                        # Uppercase
F.lower(col)                        # Lowercase
F.trim(col)                         # Remove whitespace both sides
F.ltrim(col) / F.rtrim(col)         # Remove left / right whitespace
F.concat(col1, col2, ...)           # Concatenate strings
F.concat_ws(sep, col1, col2, ...)   # Concatenate with separator
F.length(col)                       # String length
F.substring(col, pos, len)          # Substring (1-indexed)
F.split(col, pattern, limit=-1)     # Split into array
F.regexp_replace(col, pattern, rep) # Replace via regex
F.regexp_extract(col, pattern, idx) # Extract via regex group
F.like(col, pattern)                # SQL LIKE match
F.rlike(col, pattern)               # Regex match
F.initcap(col)                      # Title-case first letter
F.lpad(col, len, pad)               # Left-pad to length
F.rpad(col, len, pad)               # Right-pad to length
F.translate(col, matching, replace) # Character-by-character replace
F.format_string(fmt, *cols)         # printf-style formatting
F.instr(col, substr)                # Find position of substring
F.locate(substr, col, pos=1)        # Locate substring starting at pos
F.sha2(col, numBits)                # SHA-2 hash (256, 384, 512)
F.md5(col)                          # MD5 hash
F.encode(col, charset)              # Encode string to binary
F.decode(col, charset)              # Decode binary to string
```

### 6.2 Numeric & Math Functions

```python
F.abs(col)                          # Absolute value
F.round(col, scale=0)               # Round to scale decimal places
F.ceil(col) / F.floor(col)          # Ceiling / floor
F.sqrt(col)                         # Square root
F.pow(col, exponent)                # Power
F.log(col) / F.log2(col) / F.log10(col) # Logarithms
F.exp(col)                          # e^x
F.greatest(*cols)                   # Max value across multiple columns
F.least(*cols)                      # Min value across multiple columns
F.rand(seed)                        # Random double in [0, 1)
F.randn(seed)                       # Random Gaussian
F.bround(col, scale)                # Banker's rounding
F.signum(col)                       # Sign: -1, 0, 1
F.factorial(col)                    # Factorial
F.conv(col, fromBase, toBase)       # Base conversion
F.hex(col) / F.unhex(col)           # Hex encode/decode
F.pmod(col, divisor)                # Positive modulo
F.shiftleft(col, numBits)           # Bitwise left shift
F.shiftright(col, numBits)          # Bitwise right shift
```

### 6.3 Date & Timestamp Functions

```python
F.current_date()                    # Current date (DateType)
F.current_timestamp()               # Current timestamp
F.now()                             # Alias for current_timestamp
F.to_date(col, format)              # Parse string to DateType
F.to_timestamp(col, format)         # Parse string to TimestampType
F.date_format(col, format)          # Format date/timestamp to string
F.year(col)                         # Extract year
F.month(col)                        # Extract month (1-12)
F.dayofmonth(col)                   # Day of month (1-31)
F.dayofweek(col)                    # Day of week (1=Sunday)
F.dayofyear(col)                    # Day of year (1-366)
F.hour(col) / F.minute(col) / F.second(col)  # Time parts
F.weekofyear(col)                   # Week number
F.quarter(col)                      # Quarter (1-4)
F.trunc(col, format)                # Truncate to 'year', 'month', 'week', 'day'
F.date_trunc(format, col)           # Truncate timestamp to unit
F.date_add(col, days)               # Add days
F.date_sub(col, days)               # Subtract days
F.add_months(col, months)           # Add months
F.months_between(date1, date2)      # Months between dates (float)
F.datediff(end, start)              # Days between dates (int)
F.last_day(col)                     # Last day of month
F.next_day(col, dayOfWeek)          # Next occurrence of weekday
F.unix_timestamp(col, format)       # Convert to Unix epoch seconds
F.from_unixtime(col, format)        # Unix epoch to formatted string
F.from_utc_timestamp(col, tz)       # UTC to local timezone
F.to_utc_timestamp(col, tz)         # Local timezone to UTC
F.make_date(year, month, day)       # Construct DateType from parts
```

### 6.4 Conditional & Control Flow Functions

```python
# CASE WHEN ... THEN ... ELSE ... END equivalent
F.when(condition, value).when(...).otherwise(default)

# Null handling
F.coalesce(col1, col2, ...)         # First non-null value
F.isnull(col)                       # Boolean: is null?
F.isnan(col)                        # Boolean: is NaN?
F.nullif(col, value)                # Return null if col == value
F.ifnull(col, replacement)          # Return replacement if null
F.nvl(col, replacement)             # Alias for ifnull
F.nvl2(col, val_not_null, val_null) # If not null: val_not_null, else val_null

# Other conditionals
F.expr(sql_string)                  # Evaluate arbitrary SQL expression
F.lit(value)                        # Create a column from a literal
F.col(name)                         # Reference a column by name
F.column(name)                      # Alias for col()
```

### 6.5 Aggregation Functions

```python
F.count(col)                        # Count non-null values
F.count("*")                        # Count all rows
F.countDistinct(col1, col2, ...)    # Count distinct combinations
F.approx_count_distinct(col, rsd)   # Approximate distinct count (faster!)
F.sum(col)                          # Sum
F.sum_distinct(col)                 # Sum distinct values
F.avg(col) / F.mean(col)            # Average
F.max(col) / F.min(col)             # Max / min
F.first(col, ignorenulls=False)     # First value in group
F.last(col, ignorenulls=False)      # Last value in group
F.collect_list(col)                 # Collect all values into array (preserves duplicates)
F.collect_set(col)                  # Collect distinct values into array
F.variance(col) / F.var_pop(col)    # Sample / population variance
F.stddev(col) / F.stddev_pop(col)   # Sample / population std deviation
F.corr(col1, col2)                  # Pearson correlation
F.covar_pop(col1, col2)             # Population covariance
F.percentile_approx(col, pct, acc)  # Approximate percentile
F.skewness(col)                     # Skewness
F.kurtosis(col)                     # Kurtosis
F.product(col)                      # Product of all values
F.median(col)                       # Exact median (Spark 3.4+)
F.mode(col)                         # Mode (most frequent value, Spark 3.4+)
```

### 6.6 Array & Map Functions

```python
# Array creation and manipulation
F.array(col1, col2, ...)            # Create array from columns
F.array_contains(col, value)        # Check if element exists
F.array_distinct(col)               # Remove duplicates from array
F.array_except(col1, col2)          # Elements in col1 but not col2
F.array_intersect(col1, col2)       # Common elements
F.array_union(col1, col2)           # Union of two arrays
F.array_join(col, delimiter)        # Join array elements to string
F.array_max(col) / F.array_min(col) # Max/min element
F.array_remove(col, element)        # Remove element from array
F.array_sort(col)                   # Sort array ascending
F.array_position(col, value)        # 1-indexed position of element
F.array_append(col, element)        # Append element (Spark 3.4+)
F.array_prepend(col, element)       # Prepend element (Spark 3.4+)
F.arrays_zip(*cols)                 # Zip multiple arrays into array of structs
F.flatten(col)                      # Flatten nested arrays
F.explode(col)                      # Explode array to rows (one row per element)
F.explode_outer(col)                # Like explode but keeps null arrays as one null row
F.posexplode(col)                   # Explode with index position
F.size(col)                         # Size of array or map
F.sort_array(col, asc=True)         # Sort array
F.slice(col, start, length)         # Array slice

# Map functions
F.create_map(key1, val1, key2, val2)  # Create map from key-value columns
F.map_keys(col)                       # Extract keys as array
F.map_values(col)                     # Extract values as array
F.map_contains_key(col, key)          # Check key existence
F.map_from_arrays(keys_col, vals_col) # Build map from key/value arrays
F.map_concat(*cols)                   # Merge multiple maps
F.map_filter(col, lambda k, v: ...)   # Filter map entries
F.element_at(col, key_or_index)       # Access by key or index
F.get(col, key)                       # Safe map/array access (returns null on miss)
```

### 6.7 JSON & Struct Functions

```python
# Struct operations
F.struct(col1, col2, ...)           # Create struct from columns
F.col("struct_col.field")           # Access nested field using dot notation
F.col("struct_col")["field"]        # Access field using bracket notation

# JSON operations
F.to_json(col)                      # Convert struct/array/map to JSON string
F.from_json(col, schema)            # Parse JSON string to struct
F.get_json_object(col, path)        # Extract single value: "$.user.name"
F.json_tuple(col, *fields)          # Extract multiple fields in one pass
F.schema_of_json(json_string)       # Infer schema from JSON string literal

# Example: parsing a JSON column
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
event_schema = StructType([
    StructField("user_id", StringType()),
    StructField("action",  StringType()),
    StructField("ts",      IntegerType()),
])
df = df.withColumn("event", F.from_json(F.col("event_json"), event_schema))
df = df.select("event.user_id", "event.action", "event.ts")
```

### 6.8 Utility & ID Functions

```python
F.monotonically_increasing_id()     # Unique 64-bit ID per row (not sequential)
F.spark_partition_id()              # Partition number the row belongs to
F.input_file_name()                 # Source file path of each row
F.hash(*cols)                       # Murmur3 hash (for partitioning)
F.xxhash64(*cols)                   # xxHash64 (faster than hash())
F.crc32(col)                        # CRC32 checksum
F.uuid()                            # Generate UUID v4 string (Spark 3.5+)
```

---

## 7. Schema Management & Data Types

### 7.1 Defining Schemas Explicitly

**Always define schemas explicitly in production.** `inferSchema=True` triggers a full scan of the file, doubling read time.

```python
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType, DoubleType, FloatType,
    BooleanType, DateType, TimestampType, DecimalType,
    ArrayType, MapType, BinaryType, NullType
)

schema = StructType([
    StructField("transaction_id",  StringType(),          nullable=False),
    StructField("user_id",         LongType(),             nullable=False),
    StructField("amount",          DecimalType(18, 4),     nullable=True),
    StructField("currency",        StringType(),           nullable=True),
    StructField("event_ts",        TimestampType(),        nullable=True),
    StructField("tags",            ArrayType(StringType()), nullable=True),
    StructField("metadata",        MapType(StringType(), StringType()), nullable=True),
    StructField("address", StructType([
        StructField("city",    StringType(), nullable=True),
        StructField("country", StringType(), nullable=True),
    ]), nullable=True),
])

df = spark.read.schema(schema).json("s3://bucket/events/")
```

### 7.2 Schema Evolution

```python
# Read with schema merging (for Parquet/Delta)
df = spark.read \
    .option("mergeSchema", "true") \
    .format("parquet").load("s3://bucket/data/")

# Get DDL string (useful for documentation)
print(df.schema.simpleString())
print(df.schema.json())   # JSON representation
```

---

## 8. Handling Null Values

Nulls behave differently in Spark than in Python: `None` in Python maps to `null` in Spark. The `df.na` sub-namespace groups all null handling methods.

```python
# Drop rows with any null
df.na.drop()
df.dropna(how="any")

# Drop rows where ALL columns are null
df.na.drop(how="all")
df.dropna(how="all")

# Drop rows where specific columns are null
df.na.drop(subset=["user_id", "event_ts"])
df.dropna(thresh=3)         # Keep rows with at least 3 non-null values

# Fill nulls
df.na.fill(0)                                      # All numeric nulls → 0
df.na.fill("UNKNOWN")                              # All string nulls → "UNKNOWN"
df.fillna({"salary": 0, "dept": "UNKNOWN"})        # Column-specific fills
df.na.fill({"amount": 0.0, "currency": "USD"})

# Replace specific values (including null)
df.na.replace(["", "N/A"], None, subset=["email"])  # Coerce empty strings to null
df.na.replace([0], [None], subset=["user_id"])

# Null-safe equals (handles null comparison correctly)
df.filter(F.col("status").eqNullSafe("active"))    # True for "active", False for null

# Handle in expressions
df.withColumn("display_name",
    F.coalesce(F.col("preferred_name"), F.col("full_name"), F.lit("Anonymous"))
)
```

---

## 9. Join Strategies & Optimization

Joins are the most common source of **shuffle bottlenecks**. Choosing the right strategy is critical.

### 9.1 Join Types

```python
# Syntax: df1.join(df2, on=condition, how=join_type)

# Inner join (default)
df1.join(df2, on="user_id", how="inner")

# Left outer join
df1.join(df2, on="user_id", how="left")       # or "left_outer"

# Right outer join
df1.join(df2, on="user_id", how="right")

# Full outer join
df1.join(df2, on="user_id", how="outer")      # or "full", "full_outer"

# Left semi join — keeps df1 rows that HAVE a match in df2 (no df2 columns)
df1.join(df2, on="user_id", how="leftsemi")

# Left anti join — keeps df1 rows that DO NOT HAVE a match in df2
df1.join(df2, on="user_id", how="leftanti")

# Cross join — cartesian product (use with extreme caution!)
df1.crossJoin(df2)

# Multi-column join
df1.join(df2, on=["user_id", "product_id"], how="inner")

# Complex join expression (avoids ambiguous column issue)
df1.join(df2, df1.id == df2.customer_id, how="left")
```

### 9.2 Broadcast Hash Join (Most Important Optimization)

When one table is small (< `spark.sql.autoBroadcastJoinThreshold`, default 10 MB), Spark can ship it to every Executor. The large table is read locally and joined without **any** network shuffle.

```
DEFAULT: Sort-Merge Join               BROADCAST Hash Join
─────────────────────────              ───────────────────
Node 1: [T1_part1] ─────┐             Node 1: [T1_part1] + [S_copy] → join locally
                         shuffle       Node 2: [T1_part2] + [S_copy] → join locally
Node 2: [T1_part2] ─────┤  ← EXPENSIVE Node 3: [T1_part3] + [S_copy] → join locally
                         ↓
Node 1: [S_part_A] ─────┘             No shuffle. S is broadcast to all nodes.
```

```python
import pyspark.sql.functions as F

# Force broadcast with hint (use when Spark doesn't auto-broadcast)
result = large_orders.join(
    F.broadcast(small_store_lookup),
    on="store_id",
    how="inner"
)

# Alternatively use hint() API
result = large_orders.join(
    small_store_lookup.hint("broadcast"),
    on="store_id",
    how="inner"
)

# Tune auto-broadcast threshold (set to -1 to disable)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(50 * 1024 * 1024))  # 50 MB
```

### 9.3 Join Strategy Hints

From official docs — hints override the Catalyst optimizer's choice:

| Hint | Strategy | Use When |
|---|---|---|
| `broadcast` / `BROADCAST` | Broadcast Hash Join | One side is small enough to fit in Executor memory |
| `merge` / `MERGE` | Sort-Merge Join | Both sides are large and pre-sorted |
| `shuffle_hash` / `SHUFFLE_HASH` | Shuffle Hash Join | One side is significantly smaller than the other |
| `shuffle_replicate_nl` | Shuffle & Replicate Nested Loop | No equi-join condition (e.g., range join) |

Priority when both sides have hints: `BROADCAST > MERGE > SHUFFLE_HASH > SHUFFLE_REPLICATE_NL`

### 9.4 Avoiding Ambiguous Columns After Joins

```python
# Problem: both DataFrames have a column named "id"
joined = df_orders.join(df_users, df_orders.user_id == df_users.id)
# joined.select("id")  ← AnalysisException: ambiguous reference

# Solution 1: alias the DataFrames
orders = df_orders.alias("o")
users  = df_users.alias("u")
joined = orders.join(users, F.col("o.user_id") == F.col("u.id"))
joined.select("o.order_id", "u.name", "o.amount")

# Solution 2: drop duplicate column after join
joined = df_orders.join(df_users, df_orders.user_id == df_users.id).drop(df_users.id)

# Solution 3: rename before joining
df_users_renamed = df_users.withColumnRenamed("id", "user_id")
joined = df_orders.join(df_users_renamed, on="user_id")
```

### 9.5 Bucket Joins (Advanced — Eliminate Shuffle for Repeated Joins)

If two tables are written with the same bucketing configuration, Spark can skip the shuffle entirely on the join — even on large tables.

```python
# Write table with bucketing (do this once)
df_orders.write \
    .bucketBy(200, "user_id") \
    .sortBy("user_id") \
    .saveAsTable("orders_bucketed")

df_users.write \
    .bucketBy(200, "user_id") \
    .sortBy("user_id") \
    .saveAsTable("users_bucketed")

# Subsequent joins skip shuffle entirely
spark.table("orders_bucketed").join(spark.table("users_bucketed"), on="user_id")
```

---

## 10. Aggregations & Window Functions

### 10.1 GroupBy Aggregations

```python
# Single aggregation
df.groupBy("dept").count()

# Multiple aggregations
df.groupBy("dept", "year").agg(
    F.count("*").alias("headcount"),
    F.avg("salary").alias("avg_salary"),
    F.max("salary").alias("max_salary"),
    F.sum("bonus").alias("total_bonus"),
    F.countDistinct("team_id").alias("num_teams"),
    F.collect_set("title").alias("titles")
)

# Dict-style agg (less flexible, avoid for multi-aggregation)
df.groupBy("dept").agg({"salary": "avg", "bonus": "sum"})

# Aggregation without group (whole DataFrame)
df.agg(F.count("*"), F.sum("salary"), F.max("salary"))
```

> **`reduceByKey()` vs `groupByKey()`:** Always prefer `reduceByKey()` (on RDDs) or DataFrame `groupBy().agg()` (on DataFrames) over `groupByKey()`. The latter shuffles **all raw values** to a single node before reducing; the former does a **local pre-reduction** on each node before the network shuffle, drastically reducing data transfer.

### 10.2 Window Functions

Window functions compute values across rows **related to the current row**, without collapsing the DataFrame like `groupBy` does. They are the SQL equivalent of `OVER (PARTITION BY ... ORDER BY ...)`.

```python
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Define a window specification
window_spec = Window \
    .partitionBy("dept") \
    .orderBy(F.col("salary").desc())

# Ranking functions
df.withColumn("rank",       F.rank().over(window_spec))      # gaps for ties (1,1,3)
df.withColumn("dense_rank", F.dense_rank().over(window_spec))# no gaps (1,1,2)
df.withColumn("row_number", F.row_number().over(window_spec))# unique, no ties
df.withColumn("ntile",      F.ntile(4).over(window_spec))    # quartile bucket

# Offset functions
df.withColumn("prev_salary", F.lag("salary", 1).over(window_spec))  # previous row
df.withColumn("next_salary", F.lead("salary", 1).over(window_spec)) # next row

# Aggregate over window (no grouping)
df.withColumn("dept_avg_salary", F.avg("salary").over(Window.partitionBy("dept")))
df.withColumn("dept_max_salary", F.max("salary").over(Window.partitionBy("dept")))
df.withColumn("running_total",   F.sum("amount").over(
    Window.partitionBy("user_id").orderBy("event_ts")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
))
df.withColumn("rolling_7day_avg", F.avg("daily_revenue").over(
    Window.partitionBy("region")
    .orderBy(F.col("event_date").cast("long"))
    .rangeBetween(-6 * 86400, 0)  # 7-day range in seconds
))

# Percent rank
df.withColumn("pct_rank", F.percent_rank().over(window_spec))  # 0.0 to 1.0

# Cumulative distribution
df.withColumn("cume_dist", F.cume_dist().over(window_spec))
```

### 10.3 Cube and Rollup (Multi-Level Aggregations)

```python
# ROLLUP — hierarchical subtotals (region → dept → null for total)
df.rollup("region", "dept").agg(F.sum("revenue").alias("total_revenue")).show()

# CUBE — all possible subtotals for every combination
df.cube("region", "dept", "product").agg(F.sum("revenue")).show()

# GROUPING SETS (Spark SQL only) — pick specific combinations
spark.sql("""
    SELECT region, dept, SUM(revenue)
    FROM sales
    GROUP BY GROUPING SETS ((region, dept), (region), ())
""")
```

---

## 11. Partitioning: Repartition vs Coalesce

Partition count = degree of parallelism. This directly controls how many parallel tasks run.

### 11.1 The Fundamental Rule

| Operation | Network Shuffle? | Use Case |
|---|---|---|
| `repartition(n)` | **YES** (wide dep) | **Increase** partitions, or fix data skew by specifying a column |
| `repartition(n, col)` | **YES** | Repartition by a column (co-locate same key on same node) |
| `coalesce(n)` | **NO** (narrow dep) | **Decrease** partitions cheaply — merge locally before writing |

```python
# Increase parallelism for a large join
df_large = df.repartition(400)                  # Even distribution by hash
df_large = df.repartition(400, "user_id")       # Partition by column (bucket effect)

# Reduce tiny files before writing to storage (most common use case)
df.coalesce(10).write.parquet("s3://bucket/output/")

# Repartition by range (for sorted output — e.g., writing sorted Parquet)
df.repartitionByRange(100, F.col("event_date"))
```

### 11.2 Optimal Partition Size

The official Spark guidance is **2–4 tasks per CPU core** in your cluster. Target partition sizes of **100–200 MB** in memory.

```python
# Check current partition count
print(df.rdd.getNumPartitions())

# Rule of thumb: total data size / 128MB = good starting partition count
# Example: 50 GB dataset → ~400 partitions

# After a shuffle, check partition count
spark.conf.set("spark.sql.shuffle.partitions", "400")  # default is 200
```

### 11.3 Partition Pruning (Critical for Data Lakes)

When reading partitioned tables (Hive/Delta), always filter on partition columns to avoid reading irrelevant data.

```python
# Table is partitioned by (year, month, day)
# GOOD: Partition pruning — Spark reads only matching folders
df = spark.read.parquet("s3://data/events/") \
    .filter((F.col("year") == 2026) & (F.col("month") == 5))

# BAD: Reading all data then filtering
df = spark.read.parquet("s3://data/events/")
df = df.filter(F.col("event_date") == "2026-05-17")  # Wrong column!
```

---

## 12. Caching & Persistence

### 12.1 Storage Levels

| Level | Memory | Disk | Serialized | Replicated |
|---|---|---|---|---|
| `MEMORY_AND_DISK_DESER` | ✅ (deserialized) | spill | no | no | ← default `cache()` |
| `MEMORY_ONLY` | ✅ (deserialized) | ❌ drop if full | no | no |
| `MEMORY_ONLY_SER` | ✅ (serialized) | ❌ drop if full | **yes** | no |
| `MEMORY_AND_DISK` | ✅ | ✅ spill | **yes** | no |
| `DISK_ONLY` | ❌ | ✅ | yes | no |
| `MEMORY_ONLY_2` | ✅ | ❌ | no | **yes (2x)** |
| `OFF_HEAP` | Off-heap | ❌ | yes | no |

```python
from pyspark import StorageLevel

# .cache() is shorthand for MEMORY_AND_DISK_DESER
df.cache()

# .persist() with explicit level
df.persist(StorageLevel.MEMORY_AND_DISK)
df.persist(StorageLevel.MEMORY_ONLY_SER)   # Use with Kryo for memory efficiency

# Check if cached
print(df.is_cached)      # True/False

# Release cache (ALWAYS do this when done)
df.unpersist()
spark.catalog.clearCache()  # Clear ALL cached tables
```

### 12.2 When to Cache

```python
# PATTERN: Cache a DataFrame used by 2+ downstream actions

# Step 1: Read and transform (expensive)
clean_df = (
    spark.read.format("delta").load("dbfs:/mnt/delta/raw/")
    .filter(F.col("is_valid") == True)
    .join(F.broadcast(lookup_df), on="product_id")
    .withColumn("revenue", F.col("qty") * F.col("unit_price"))
)

# Step 2: Cache before branching
clean_df.cache()

# Step 3: First action — triggers computation AND fills cache
total_revenue = clean_df.agg(F.sum("revenue")).collect()[0][0]

# Step 4: Second action — reads FROM CACHE, extremely fast
clean_df.write.format("delta").mode("overwrite").save("dbfs:/mnt/delta/gold/")

# Step 5: Third use
clean_df.groupBy("region").agg(F.sum("revenue")).write \
    .format("delta").mode("overwrite").save("dbfs:/mnt/delta/summary/")

# Step 6: RELEASE cache (important!)
clean_df.unpersist()
```

### 12.3 Caching SQL Tables

```python
# Cache using catalog API (uses in-memory columnar format with auto compression)
spark.catalog.cacheTable("my_database.fact_transactions")
spark.catalog.uncacheTable("my_database.fact_transactions")

# In SQL
spark.sql("CACHE TABLE my_database.dim_product")
spark.sql("UNCACHE TABLE my_database.dim_product")

# Lazy vs eager caching
spark.sql("CACHE LAZY TABLE my_database.dim_product")  # cached on first scan
spark.sql("CACHE TABLE my_database.dim_product")        # cached immediately
```

---

## 13. UDFs vs Built-ins — The Performance Choice

### 13.1 Why UDFs Are Slow

Python UDFs break out of the JVM into the Python interpreter for **every single row**. Each row requires:
1. Serialize the row from JVM to Python (pickle)
2. Execute the Python function
3. Deserialize the result back to JVM

The Catalyst optimizer treats a UDF as a **black box** — it cannot inspect or optimize around it.

```
Built-in Function Flow:         Python UDF Flow:
JVM                             JVM   ↔  Python Process
 └─ Catalyst Optimizer          │         │
 └─ Tungsten Codegen            Row 1:  serialize → func(row1) → deserialize
 └─ Execute in C code           Row 2:  serialize → func(row2) → deserialize
 └─ Vectorized SIMD ops         Row n:  serialize → func(rowN) → deserialize
```

### 13.2 The Decision Hierarchy

```
1. Built-in F.xxx() function              → ALWAYS prefer
2. SQL expression via F.expr()            → Try this if no direct built-in
3. Pandas UDF (Vectorized UDF)            → Use if built-in is truly insufficient
4. Python UDF (row-by-row)                → Last resort only
```

### 13.3 Replacing UDFs with Built-ins

```python
# ❌ BAD: Python UDF
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def clean_phone(phone):
    if phone:
        return phone.replace("-", "").replace(" ", "").strip()
    return None

df.withColumn("phone_clean", clean_phone(F.col("phone")))

# ✅ GOOD: Built-in equivalents
df.withColumn("phone_clean",
    F.trim(F.regexp_replace(F.col("phone"), r"[-\s]", ""))
)
```

### 13.4 Pandas UDFs (Vectorized — Much Better than Row UDFs)

When you truly need custom logic, use **Pandas UDFs** (also called Vectorized UDFs). Spark passes entire **column batches** as Pandas Series to Python — far fewer serialization round trips.

```python
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd

# Series → Series transformation
@pandas_udf(DoubleType())
def score_customer(revenue: pd.Series, tenure: pd.Series) -> pd.Series:
    return (revenue * 0.6 + tenure * 0.4) / 100.0

df.withColumn("score", score_customer(F.col("revenue"), F.col("tenure")))

# GroupBy + apply (like pandas groupby().apply())
from pyspark.sql.functions import PandasUDFType

output_schema = StructType([
    StructField("dept", StringType()),
    StructField("normalized_salary", DoubleType()),
])

@pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
def normalize_salary(df: pd.DataFrame) -> pd.DataFrame:
    df["normalized_salary"] = (df["salary"] - df["salary"].mean()) / df["salary"].std()
    return df[["dept", "normalized_salary"]]

df.groupBy("dept").apply(normalize_salary)
```

### 13.5 `F.expr()` — SQL Expressions as a Bridge

```python
# Execute any Spark SQL expression inside a DataFrame operation
df.withColumn("tax", F.expr("amount * 0.07"))
df.withColumn("full_name", F.expr("concat_ws(' ', first_name, last_name)"))
df.filter(F.expr("datediff(current_date(), hire_date) > 365"))

# Complex conditional
df.withColumn("status", F.expr("""
    CASE
        WHEN days_overdue > 90 THEN 'WRITE_OFF'
        WHEN days_overdue > 30 THEN 'PAST_DUE'
        WHEN days_overdue > 0  THEN 'LATE'
        ELSE 'CURRENT'
    END
"""))
```

---

## 14. Data I/O Best Practices

### 14.1 File Format Guide

| Format | Read Speed | Write Speed | Splittable | Schema | Best For |
|---|---|---|---|---|---|
| **Parquet** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Embedded | Analytical workloads (columnar, high compression) |
| **Delta Lake** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Schema enforcement | Databricks lakehouse, ACID transactions |
| **ORC** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Embedded | Hive-compatible columnar workloads |
| **Avro** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Separate | Kafka events, streaming, schema registry |
| **JSON** | ⭐⭐ | ⭐⭐⭐ | ✅ | Inferred/explicit | Semi-structured APIs, flexibility |
| **CSV** | ⭐ | ⭐⭐⭐ | ✅ | Must define | Legacy systems, human-readable |
| **Gzip (any)** | ⭐ | ⭐⭐ | ❌ **Non-splittable** | — | ⚠️ Avoid for large files — forces 1 partition |

> **Rule:** Use **Parquet with Snappy** (default) or **Delta Lake** for production data lakes. Never store large data as Gzip — it cannot be split into parallel partitions.

### 14.2 Reading Data

```python
# Parquet (recommended default)
df = spark.read.parquet("s3://bucket/data/year=2026/month=05/")
df = spark.read.format("parquet").load("s3://bucket/data/")

# Delta Lake (Databricks standard)
df = spark.read.format("delta").load("dbfs:/mnt/delta/transactions/")
df = spark.read.format("delta").option("versionAsOf", 5).load("...")   # Time travel
df = spark.read.format("delta").option("timestampAsOf", "2026-05-01").load("...")

# CSV with all options
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("sep", "|") \
    .option("quote", '"') \
    .option("escape", "\\") \
    .option("encoding", "UTF-8") \
    .option("nullValue", "NULL") \
    .option("nanValue", "NaN") \
    .option("multiLine", "false") \
    .option("dateFormat", "yyyy-MM-dd") \
    .option("timestampFormat", "yyyy-MM-dd'T'HH:mm:ss") \
    .schema(my_schema) \
    .load("s3://bucket/uploads/*.csv")

# JSON
df = spark.read.format("json") \
    .schema(my_schema) \
    .option("multiLine", "false") \   # true for pretty-printed JSON
    .load("s3://bucket/events/")

# JDBC (relational database)
df = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://host:5432/dbname") \
    .option("dbtable", "schema.table") \
    .option("user", "username") \
    .option("password", "password") \
    .option("numPartitions", 20) \
    .option("partitionColumn", "id") \
    .option("lowerBound", "1") \
    .option("upperBound", "10000000") \
    .load()
```

### 14.3 Writing Data

```python
# Parquet (standard write)
df.write \
    .mode("overwrite") \           # overwrite | append | error | ignore
    .option("compression", "snappy") \
    .partitionBy("year", "month") \    # creates folder hierarchy
    .parquet("s3://bucket/output/")

# Delta Lake (Databricks best practice)
df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("event_date") \
    .save("dbfs:/mnt/delta/fact_events/")

# Write to Hive/Unity Catalog table
df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("catalog.schema.table_name")

# Controlling output file count (avoid tiny files!)
df.coalesce(1).write.parquet("s3://bucket/small_result/")   # 1 file
df.repartition(20).write.parquet("s3://bucket/output/")     # 20 files

# Write with specific file naming (Databricks)
df.write.format("delta").option("maxRecordsPerFile", 1000000).save("...")
```

### 14.4 Reading Partitioned Tables Efficiently

```python
# Reading only specific date partitions from a Hive-partitioned path
df = spark.read.parquet("s3://data/events/") \
    .filter((F.col("year") == 2026) & (F.col("month") == 5))

# Use spark.table() for catalog tables — enables statistics-based optimization
df = spark.table("mydb.fact_events") \
    .where("event_date BETWEEN '2026-05-01' AND '2026-05-17'")
```

---

## 15. Adaptive Query Execution (AQE)

AQE (enabled by default since Spark 3.2.0) re-optimizes query plans **at runtime** using actual statistics collected after each shuffle stage — not just estimated statistics.

### 15.1 Key AQE Features

```python
# Enable AQE (default true in Spark 3.2+)
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

**Feature 1: Post-Shuffle Partition Coalescing**
Automatically merges small shuffle partitions after a shuffle, reducing task overhead:
```python
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64mb")  # target partition size
spark.conf.set("spark.sql.adaptive.coalescePartitions.minPartitionSize", "1mb")
```

**Feature 2: Dynamic Join Strategy Switching**
Converts a Sort-Merge Join to a Broadcast Hash Join at runtime if actual data turns out to be small:
```python
spark.conf.set("spark.sql.adaptive.autoBroadcastJoinThreshold", "30mb")  # runtime threshold
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")   # local read after broadcast conversion
```

**Feature 3: Skew Join Optimization**
Automatically detects and splits skewed partitions in Sort-Merge Joins:
```python
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5.0")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256mb")
```

---

## 16. Shuffle Internals & Optimization

The shuffle is **the most expensive operation in Spark**. Every wide dependency triggers:
1. **Shuffle Write:** Map tasks write output data to local disk, sorted/hashed by target partition
2. **Shuffle Read:** Reduce tasks fetch their partition's data from all map nodes over the network
3. **Sort & Merge:** CPU processes the fetched data (sort, hash, aggregate)

### 16.1 Minimizing Shuffle Impact

```python
# 1. Push filters BEFORE joins (Catalyst does this automatically, but make it explicit)
# BAD
result = df_orders.join(df_users, "user_id").filter(F.col("region") == "APAC")

# GOOD (filter reduces data before shuffle)
result = df_orders.filter(F.col("region") == "APAC").join(df_users, "user_id")

# 2. Use reduceByKey/agg instead of groupByKey
# BAD (on RDDs): sends all raw values over network
rdd.groupByKey().mapValues(sum)

# GOOD: pre-aggregates locally before shuffle
rdd.reduceByKey(lambda a, b: a + b)

# 3. Avoid orderBy() unless truly necessary (always global shuffle)
# Use window functions with partitionBy for per-group ordering instead
window = Window.partitionBy("dept").orderBy(F.col("salary").desc())
df.withColumn("rank", F.rank().over(window))  # no global shuffle!

# 4. Tune shuffle partitions based on data size
# Default is 200 — often too low for large data, too high for small data
spark.conf.set("spark.sql.shuffle.partitions", "400")
```

### 16.2 Shuffle Configurations

```python
# Core shuffle partition tuning
spark.conf.set("spark.sql.shuffle.partitions", "200")  # Default; tune per job size

# Enable sort-based shuffle (default since Spark 2.0, most efficient)
spark.conf.set("spark.shuffle.manager", "sort")

# Enable off-heap memory for shuffle sort (Tungsten)
spark.conf.set("spark.memory.offHeap.enabled", "true")
spark.conf.set("spark.memory.offHeap.size", "4g")     # Off-heap allocation per Executor

# Spill compression (reduces disk I/O during spills)
spark.conf.set("spark.shuffle.spill.compress", "true")
spark.conf.set("spark.shuffle.compress", "true")
spark.conf.set("spark.io.compression.codec", "lz4")   # lz4 is fastest decompression
```

---

## 17. Memory Management Deep Dive

### 17.1 Unified Memory Manager (UMM) Architecture

For a 5 GB Executor JVM heap:

```
┌────────────────────────────────────────────────────────────┐
│                    JVM Heap (5.0 GB)                       │
├────────────────────────────────────────────────────────────┤
│  Reserved Memory: 300 MB  (internal buffers, hardcoded)    │
├────────────────────────────────────────────────────────────┤
│  User Memory: 40% × 4.7 GB ≈ 1.88 GB                      │
│  (UDFs, user data structures, metadata, OOM safety buffer) │
├────────────────────────────────────────────────────────────┤
│  Spark Memory Pool: 60% × 4.7 GB ≈ 2.82 GB                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Execution Memory: 50% = 1.41 GB (elastic)          │◀─── borrow
│  │  (Joins, sorts, aggregation buffers, shuffle)        │──▶ evict Storage
│  ├──────────────────────────────────────────────────────┤  │
│  │  Storage Memory: 50% = 1.41 GB (elastic)            │  │
│  │  (RDD cache, broadcast vars, DataFrame cache)        │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

**Eviction rules:**
- Execution can borrow from Storage if Storage is idle.
- Storage can borrow from Execution if Execution is idle.
- **Execution can forcefully evict Storage** — active computation beats cached data.
- Storage **cannot** evict Execution blocks.

### 17.2 Memory Configuration

```python
# spark.memory.fraction = fraction of (heap - 300MB) for Spark Memory Pool
# Default: 0.6 → 60% for Spark, 40% for User Memory
spark.conf.set("spark.memory.fraction", "0.6")

# spark.memory.storageFraction = fraction of Spark Memory Pool protected for Storage
# Default: 0.5 → 50% of Spark Pool is protected minimum for Storage
spark.conf.set("spark.memory.storageFraction", "0.5")

# Executor memory (set in cluster config or spark-submit)
# --executor-memory 8g
# --driver-memory 4g
```

### 17.3 Preventing OOM Errors

```python
# 1. Never collect() large DataFrames
# BAD
all_rows = df.collect()  # 100M rows → Driver OOM

# GOOD: aggregate first, then collect
summary = df.agg(F.count("*"), F.sum("revenue")).collect()

# 2. Unpersist cached data when done
df.cache()
# ... use df ...
df.unpersist()  # Free executor memory immediately

# 3. Use .persist(MEMORY_AND_DISK) to allow disk spillage
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)

# 4. Avoid collect_list() on large groups
# BAD
df.groupBy("user_id").agg(F.collect_list("event"))  # can build massive arrays

# GOOD: use array size limit or pre-aggregate
df.groupBy("user_id").agg(F.count("event").alias("event_count"))

# 5. Monitor memory pressure in Spark UI → Executors tab → Storage Memory Used
```

---

## 18. Data Skew Detection & Remediation

Data skew occurs when one partition has **much more data** than others — one Executor does all the work while others idle. This causes jobs to seemingly "hang" on the last few tasks.

### 18.1 Detecting Skew

In the **Spark UI → Stages Tab**:
- Sort tasks by **Duration**
- If `Max` duration >> `75th percentile` duration → **skew exists**
- Check the **Input Size** column for the outlier tasks

```python
# Diagnose skew programmatically
df.groupBy("join_key").count().orderBy(F.col("count").desc()).show(20)

# Check partition sizes (sample approach)
df.withColumn("partition_id", F.spark_partition_id()) \
  .groupBy("partition_id").count() \
  .orderBy(F.col("count").desc()).show(20)
```

### 18.2 Remediation Strategies

**Strategy 1: AQE Skew Join (automatic, Spark 3.0+)**
```python
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5.0")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256mb")
```

**Strategy 2: Salting** (manual, for pre-AQE or extreme skew)
```python
import random

# Add random salt to skewed key before joining
NUM_SALT_BUCKETS = 10

# Salt the large table
df_large_salted = df_large.withColumn(
    "salted_key",
    F.concat(F.col("user_id"), F.lit("_"), (F.rand() * NUM_SALT_BUCKETS).cast("int").cast("string"))
)

# Explode the small table to match all salt values
df_small_exploded = df_small.withColumn(
    "salt_bucket", F.explode(F.array([F.lit(i) for i in range(NUM_SALT_BUCKETS)]))
).withColumn(
    "salted_key",
    F.concat(F.col("user_id"), F.lit("_"), F.col("salt_bucket").cast("string"))
)

# Join on salted key
result = df_large_salted.join(df_small_exploded, on="salted_key") \
    .drop("salted_key", "salt_bucket")
```

**Strategy 3: Isolate the Skewed Key**
```python
# Separate skewed and non-skewed rows, join differently, union
skewed_key = "user_id_12345"

df_skewed     = df_large.filter(F.col("user_id") == skewed_key)
df_non_skewed = df_large.filter(F.col("user_id") != skewed_key)

# Non-skewed: normal join
result_normal = df_non_skewed.join(df_small, on="user_id")

# Skewed key: broadcast join for this key
df_small_key = df_small.filter(F.col("user_id") == skewed_key)
result_skewed = df_skewed.join(F.broadcast(df_small_key), on="user_id")

# Combine
result = result_normal.union(result_skewed)
```

---

## 19. Shared Variables: Broadcast & Accumulators

### 19.1 Broadcast Variables

Broadcast variables efficiently distribute large read-only lookup tables to all Executors — stored **once** in each Executor's memory rather than once per Task.

```python
# Create broadcast variable from driver (e.g., a Python dict or list)
country_codes = {"US": "United States", "UK": "United Kingdom", "DE": "Germany"}
bc_countries = sc.broadcast(country_codes)

# Use inside a UDF (accessing via .value)
@udf(StringType())
def get_country_name(code):
    return bc_countries.value.get(code, "Unknown")

df.withColumn("country_name", get_country_name(F.col("country_code")))

# Destroy when done to free Executor memory
bc_countries.destroy()

# For DataFrames: use F.broadcast() hint instead (Catalyst-aware)
import pyspark.sql.functions as F
df.join(F.broadcast(small_lookup), on="key")
```

### 19.2 Accumulators

Accumulators aggregate values from Executors back to the Driver — useful for counting events, errors, or metrics during transformations.

```python
# Built-in accumulator (long counter)
error_count = sc.accumulator(0)
null_count  = sc.accumulator(0)

def validate_row(row):
    if row.email is None:
        null_count.add(1)
    if row.amount < 0:
        error_count.add(1)
    return row

df.foreach(validate_row)

print(f"Null emails: {null_count.value}")
print(f"Negative amounts: {error_count.value}")

# Custom accumulator (aggregate sets, lists, etc.)
from pyspark import AccumulatorParam

class SetAccumulator(AccumulatorParam):
    def zero(self, value):
        return set()
    def addInPlace(self, val1, val2):
        return val1 | val2

seen_errors = sc.accumulator(set(), SetAccumulator())
```

> ⚠️ **Accumulator Warning:** Accumulators inside transformations may be **double-counted** if tasks are retried (fault tolerance re-runs). Only rely on accumulator values after a completed action, and only use them for metrics — not for data correctness logic.

---

## 20. Spark UI & Monitoring

The Spark UI is your primary diagnostic tool. Access it via the Driver URL (usually port 4040) or in Databricks through the cluster's "Spark UI" button.

### 20.1 Key Tabs and What to Look For

**SQL / DataFrame Tab:**
| Indicator | What It Means | Action |
|---|---|---|
| `WholeStageCodegen` wrapper | Tungsten is active — optimal execution | Nothing |
| `HashAggregate` without Codegen | Optimization failed | Check config, reduce complexity |
| `Exchange` node | A shuffle is happening | Quantify bytes; consider broadcast |
| `BroadcastHashJoin` | Small table was broadcast | Good — confirms optimization |
| `SortMergeJoin` | Both sides were shuffled | Consider broadcast if one side < threshold |

**Stages Tab:**
| Check | Problem Signal | Fix |
|---|---|---|
| Task Duration: Max >> 75th pct | **Data skew** | Salting, AQE skew hints |
| GC Time > 10% of task time | **Too many Java objects** | Use serialized storage, tune `-Xmn` |
| Shuffle Read Bytes very large | **Expensive shuffle** | Broadcast join, reduce data before shuffle |
| Tasks with `FAILED` status | Node failures / OOM | Check Executor logs, increase memory |

**Executors Tab:**
| Check | Problem Signal | Fix |
|---|---|---|
| Storage Memory near 100% | Cache pressure | Unpersist unused DataFrames |
| GC Time column high | JVM GC overload | `persist(MEMORY_AND_DISK)`, reduce cache |
| One Executor handling most tasks | **Skew** | See Section 18 |

### 20.2 Programmatic Plan Inspection

```python
# Quick physical plan
df.explain()

# Full extended plan: Parsed → Analyzed → Optimized → Physical
df.explain("extended")

# Show cost estimates (requires ANALYZE TABLE)
df.explain("cost")

# Formatted tree view (most readable, Spark 3.0+)
df.explain("formatted")

# Run ANALYZE TABLE to collect statistics for the optimizer
spark.sql("ANALYZE TABLE mydb.fact_orders COMPUTE STATISTICS")
spark.sql("ANALYZE TABLE mydb.fact_orders COMPUTE STATISTICS FOR COLUMNS user_id, event_date, amount")
```

---

## 21. Databricks-Specific Best Practices

### 21.1 Delta Lake (Databricks Native Format)

```python
# Write Delta table
df.write.format("delta").mode("overwrite").save("dbfs:/mnt/delta/table/")
df.write.format("delta").mode("append").saveAsTable("catalog.db.table")

# MERGE (upsert) — ACID transaction
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "dbfs:/mnt/delta/customers/")

delta_table.alias("target").merge(
    source=df_updates.alias("source"),
    condition="target.customer_id = source.customer_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# MERGE with conditions
delta_table.alias("t").merge(
    df_updates.alias("s"),
    "t.id = s.id"
).whenMatchedUpdate(
    condition="s.version > t.version",
    set={"value": "s.value", "updated_at": "s.updated_at"}
).whenNotMatchedInsert(
    values={"id": "s.id", "value": "s.value", "updated_at": "s.updated_at"}
).whenNotMatchedBySourceDelete() \  # Delete unmatched target rows
.execute()

# Time travel
df = spark.read.format("delta").option("versionAsOf", 10).load("dbfs:/mnt/delta/table/")
df = spark.read.format("delta").option("timestampAsOf", "2026-01-01").load("...")

# OPTIMIZE (compact small files) + ZORDER (co-locate by column)
spark.sql("OPTIMIZE catalog.db.fact_events ZORDER BY (user_id, event_date)")

# VACUUM (remove old files)
spark.sql("VACUUM catalog.db.fact_events RETAIN 168 HOURS")  # keep 7 days

# View table history
spark.sql("DESCRIBE HISTORY catalog.db.fact_events").show()

# Schema evolution (auto-merge new columns)
df.write.format("delta").option("mergeSchema", "true").mode("append").save("...")
```

### 21.2 Databricks Runtime Optimizations

```python
# Photon Engine: Use SQL/DataFrame API — Photon accelerates Spark SQL operations
# Photon is enabled at the cluster level; no code change needed

# Auto Optimize (auto-compact small files on write)
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

# Predictive I/O (Databricks-specific read optimization)
# Enabled automatically on photon clusters

# Cluster configuration best practices:
# - Use auto-scaling clusters for variable workloads
# - Use fixed clusters for SLA-sensitive streaming
# - Enable spot/preemptible workers for cost savings on batch
```

### 21.3 Unity Catalog Data Access

```python
# Read with 3-level namespace: catalog.schema.table
df = spark.table("main.sales.fact_transactions")
df = spark.sql("SELECT * FROM main.sales.fact_transactions WHERE dt = '2026-05-17'")

# Set default catalog and schema for session
spark.catalog.setCurrentCatalog("main")
spark.catalog.setCurrentDatabase("sales")

# List tables
spark.catalog.listTables("main.sales")

# Check table properties
spark.sql("DESCRIBE EXTENDED main.sales.fact_transactions").show(50, False)
```

---

## 22. Key Configuration Reference

### 22.1 Essential SparkConf Parameters

```python
spark = SparkSession.builder \
    .appName("ProductionPipeline") \
    # ─── Performance ───────────────────────────────────────────────────────
    .config("spark.sql.adaptive.enabled",                 "true") \   # AQE
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled",        "true") \
    .config("spark.sql.shuffle.partitions",               "400") \    # tune per job
    .config("spark.sql.autoBroadcastJoinThreshold",       "10m") \    # auto broadcast < 10MB
    .config("spark.sql.broadcastTimeout",                 "600") \    # seconds
    # ─── Memory ─────────────────────────────────────────────────────────────
    .config("spark.memory.fraction",                      "0.6") \    # 60% to Spark Pool
    .config("spark.memory.storageFraction",               "0.5") \    # 50% of Pool to Storage
    .config("spark.memory.offHeap.enabled",               "true") \   # Tungsten off-heap
    .config("spark.memory.offHeap.size",                  "2g") \
    # ─── Serialization ──────────────────────────────────────────────────────
    .config("spark.serializer",          "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \ # compress cache
    .config("spark.sql.inMemoryColumnarStorage.batchSize", "10000") \ # columnar batch
    # ─── I/O ────────────────────────────────────────────────────────────────
    .config("spark.sql.files.maxPartitionBytes",          "134217728") \ # 128 MB per partition
    .config("spark.shuffle.compress",                     "true") \
    .config("spark.shuffle.spill.compress",               "true") \
    .config("spark.io.compression.codec",                 "lz4") \    # lz4 or zstd
    # ─── Dynamic Allocation ─────────────────────────────────────────────────
    .config("spark.dynamicAllocation.enabled",            "true") \
    .config("spark.dynamicAllocation.minExecutors",       "2") \
    .config("spark.dynamicAllocation.maxExecutors",       "50") \
    .config("spark.dynamicAllocation.executorIdleTimeout","60s") \
    .config("spark.dynamicAllocation.shuffleTracking.enabled", "true") \
    .getOrCreate()
```

### 22.2 Quick Configuration Cheat Sheet

| Config Key | Default | Recommended | When to Change |
|---|---|---|---|
| `spark.sql.shuffle.partitions` | `200` | `2-4x num_cores` | Large shuffles stall; too many tiny tasks |
| `spark.sql.autoBroadcastJoinThreshold` | `10mb` | `30mb–200mb` | Small dims not being broadcast |
| `spark.sql.adaptive.enabled` | `true` | `true` | Already default in 3.2+ |
| `spark.sql.adaptive.advisoryPartitionSizeInBytes` | `64mb` | `64mb–128mb` | AQE coalescing target size |
| `spark.memory.fraction` | `0.6` | `0.6` | GC pressure high → lower to 0.5 |
| `spark.memory.storageFraction` | `0.5` | `0.5` | Heavy caching → increase; heavy joins → decrease |
| `spark.serializer` | Java | Kryo | Always set to Kryo for performance |
| `spark.sql.files.maxPartitionBytes` | `128mb` | `128mb–256mb` | Small files problem |
| `spark.dynamicAllocation.enabled` | cluster-dep | `true` | Multi-user/multi-job environments |

---

## 23. Complete Production Pipeline Example

This example demonstrates a full ETL pipeline incorporating all the best practices from this guide.

```python
"""
Production PySpark Data Pipeline
Demonstrates: Schema-first reading, broadcast join, AQE,
              window functions, caching, Delta write, proper null handling
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType,
    DoubleType, TimestampType, IntegerType, DateType
)

# ── 1. Session Initialization ───────────────────────────────────────────────

spark = (SparkSession.builder
    .appName("SalesAggregationPipeline")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.skewJoin.enabled", "true")
    .config("spark.sql.shuffle.partitions", "400")
    .config("spark.sql.autoBroadcastJoinThreshold", str(50 * 1024 * 1024))  # 50 MB
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate())

sc = spark.sparkContext
sc.setLogLevel("WARN")

# ── 2. Define Schemas Explicitly (never use inferSchema in production) ───────

orders_schema = StructType([
    StructField("order_id",    StringType(),    nullable=False),
    StructField("user_id",     LongType(),      nullable=False),
    StructField("product_id",  StringType(),    nullable=False),
    StructField("qty",         IntegerType(),   nullable=True),
    StructField("unit_price",  DoubleType(),    nullable=True),
    StructField("event_ts",    TimestampType(), nullable=True),
    StructField("region",      StringType(),    nullable=True),
    StructField("is_cancelled",StringType(),    nullable=True),
])

products_schema = StructType([
    StructField("product_id",  StringType(), nullable=False),
    StructField("product_name",StringType(), nullable=True),
    StructField("category",    StringType(), nullable=True),
    StructField("brand",       StringType(), nullable=True),
])

# ── 3. Read Source Data ──────────────────────────────────────────────────────

# Large fact table — read with partition pruning
df_orders = (spark.read
    .schema(orders_schema)
    .format("delta")
    .load("dbfs:/mnt/delta/raw/orders/")
    .filter(
        (F.col("event_ts") >= F.lit("2026-01-01").cast(TimestampType())) &
        (F.col("event_ts") <  F.lit("2026-06-01").cast(TimestampType()))
    )
)

# Small lookup table — will be broadcast
df_products = (spark.read
    .schema(products_schema)
    .format("delta")
    .load("dbfs:/mnt/delta/raw/products/")
)

print(f"Orders partitions: {df_orders.rdd.getNumPartitions()}")

# ── 4. Data Quality & Cleaning ───────────────────────────────────────────────

df_orders_clean = (df_orders
    # Drop rows with critical null values
    .dropna(subset=["order_id", "user_id", "product_id"])

    # Normalize string fields
    .withColumn("region",       F.upper(F.trim(F.col("region"))))
    .withColumn("is_cancelled", F.lower(F.trim(F.col("is_cancelled"))))

    # Coerce invalid strings to proper null
    .withColumn("region", F.when(F.col("region").isin("", "NULL", "N/A"), None)
                           .otherwise(F.col("region")))

    # Fill remaining nulls
    .fillna({"qty": 1, "region": "UNKNOWN"})

    # Derive columns using built-ins only (no UDFs)
    .withColumn("revenue",    F.round(F.col("qty") * F.col("unit_price"), 4))
    .withColumn("event_date", F.to_date(F.col("event_ts")))
    .withColumn("event_year", F.year(F.col("event_ts")))
    .withColumn("event_month",F.month(F.col("event_ts")))

    # Business logic via when/otherwise
    .withColumn("is_active",
        F.when(F.col("is_cancelled").isin("true", "yes", "1"), False)
         .otherwise(True))

    # Drop cancelled orders
    .filter(F.col("is_active") == True)
    .drop("is_cancelled", "is_active")
)

# ── 5. Enrich with Lookup Table (Broadcast Join) ─────────────────────────────

df_enriched = df_orders_clean.join(
    F.broadcast(df_products),  # Products is small → broadcast, no shuffle
    on="product_id",
    how="left"                 # Keep orders even if product not in lookup
)

# ── 6. Cache Before Multiple Downstream Uses ─────────────────────────────────

df_enriched.cache()
print(f"Enriched row count: {df_enriched.count()}")  # First action → fills cache

# ── 7. Aggregations ──────────────────────────────────────────────────────────

# Monthly revenue by region + category
df_monthly = (df_enriched
    .groupBy("event_year", "event_month", "region", "category")
    .agg(
        F.count("order_id").alias("order_count"),
        F.countDistinct("user_id").alias("unique_users"),
        F.sum("revenue").alias("total_revenue"),
        F.avg("revenue").alias("avg_order_revenue"),
        F.percentile_approx("revenue", 0.95, 1000).alias("p95_revenue"),
    )
)

# ── 8. Window Functions — Running Totals + Ranking ──────────────────────────

window_region = (Window
    .partitionBy("region")
    .orderBy("event_year", "event_month")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow))

window_rank = (Window
    .partitionBy("event_year", "event_month")
    .orderBy(F.col("total_revenue").desc()))

df_final = (df_monthly
    .withColumn("running_revenue", F.sum("total_revenue").over(window_region))
    .withColumn("month_rank_in_period", F.rank().over(window_rank))
    .withColumn("pct_of_period_revenue",
        F.round(
            F.col("total_revenue") / F.sum("total_revenue").over(
                Window.partitionBy("event_year", "event_month")
            ) * 100, 2
        )
    )
)

# ── 9. Top Users (another downstream use — reads from cache) ─────────────────

window_user = Window.partitionBy("user_id").orderBy(F.col("event_ts").desc())

df_user_stats = (df_enriched
    .groupBy("user_id")
    .agg(
        F.count("order_id").alias("total_orders"),
        F.sum("revenue").alias("lifetime_value"),
        F.max("event_date").alias("last_order_date"),
    )
    .withColumn("tier",
        F.when(F.col("lifetime_value") >= 10000, "PLATINUM")
         .when(F.col("lifetime_value") >= 5000,  "GOLD")
         .when(F.col("lifetime_value") >= 1000,  "SILVER")
         .otherwise("BRONZE")
    )
)

# ── 10. Write Outputs (Delta format, partitioned for efficient reads) ─────────

(df_final
    .repartition(20, "region")     # co-locate by region before write
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("event_year", "event_month")
    .saveAsTable("main.analytics.monthly_revenue_summary")
)

(df_user_stats
    .coalesce(10)                  # reduce file count for small result
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.analytics.user_lifetime_stats")
)

# ── 11. Release Cache ─────────────────────────────────────────────────────────

df_enriched.unpersist()

# ── 12. Post-write Optimization (Delta OPTIMIZE + ZORDER) ────────────────────

spark.sql("""
    OPTIMIZE main.analytics.monthly_revenue_summary
    ZORDER BY (region, event_year, event_month)
""")

print("Pipeline complete.")
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PySpark Best Practices — Quick Reference                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  API CHOICE                                                                 │
│  ✅ DataFrame / Spark SQL (Catalyst-optimizable)                             │
│  ⚠️  Pandas UDF (vectorized, use when built-in is insufficient)              │
│  ❌ Python UDF (row-by-row, JVM↔Python serialization per row)               │
│  ❌ Raw RDD with Python lambda (bypasses Catalyst entirely)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  JOINS                                                                      │
│  ✅ F.broadcast(small_df)  — eliminates shuffle for small tables            │
│  ✅ AQE auto-broadcast     — runtime detection (spark.sql.adaptive=true)    │
│  ✅ Bucket joins           — pre-partition by join key, no runtime shuffle  │
│  ❌ Sort-merge join on skewed keys — use salting or AQE skew hints          │
├─────────────────────────────────────────────────────────────────────────────┤
│  PARTITIONS                                                                 │
│  ✅ coalesce(n)  — decrease partitions, NO shuffle (narrow dep)             │
│  ✅ repartition(n, col) — increase or fix skew, triggers shuffle             │
│  ✅ Filter BEFORE join — push predicates close to source                    │
│  ❌ repartition() just to decrease count — use coalesce() instead           │
├─────────────────────────────────────────────────────────────────────────────┤
│  CACHING                                                                    │
│  ✅ cache() / persist() when DataFrame used 2+ times                        │
│  ✅ Always unpersist() when done                                             │
│  ❌ Cache every intermediate result — wastes Executor memory                │
├─────────────────────────────────────────────────────────────────────────────┤
│  DATA FORMATS                                                               │
│  ✅ Parquet + Snappy  — columnar, splittable, compressed                    │
│  ✅ Delta Lake        — ACID, schema enforcement, time travel               │
│  ❌ Gzip              — non-splittable → forces 1 partition                 │
│  ❌ CSV without schema — inferSchema forces extra full scan                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  PERFORMANCE TRAPS                                                          │
│  ❌ collect() on large data — OOM on Driver                                 │
│  ❌ orderBy() mid-pipeline — global shuffle; push to end or use windows     │
│  ❌ groupByKey() on RDDs  — use reduceByKey() instead                       │
│  ❌ distinct() without need — always a shuffle; deduplicate intentionally   │
│  ❌ Multiple withColumn() — chain into one select() for fewer plan nodes    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Sources: [Apache Spark Official Documentation](https://spark.apache.org/docs/latest/) · [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/reference/) · Apache Spark RDD Paper (Zaharia et al.) · Spark SQL Performance Tuning Guide · Databricks Documentation*
