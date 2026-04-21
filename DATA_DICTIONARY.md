# Data Dictionary — DDC CWICR

This document describes the physical layout, column semantics,
and provenance of the DDC CWICR dataset (Construction Work Items,
Costs & Resources). Licensing of the dataset is documented in
[LICENSE](./LICENSE) and [NOTICE](./NOTICE); this file describes
the *data* itself.

## Dataset shape (as of April 2026)

- **55,719** work items / rates
- **27,672** unique resources (labour / material / equipment)
- **11** language × region tracks
- **85** columns in the primary work-items tables

## Directory layout

Each language / region track has its own directory:

| Directory               | Language                          | Reference region     |
|-------------------------|-----------------------------------|----------------------|
| `AR___DDC_CWICR/`       | Arabic                            | Dubai (UAE)          |
| `DE___DDC_CWICR/`       | German                            | Germany              |
| `EN___DDC_CWICR/`       | English                           | Toronto (Canada)     |
| `ES___DDC_CWICR/`       | Spanish                           | Spain                |
| `FR___DDC_CWICR/`       | French                            | France               |
| `HI___DDC_CWICR/`       | Hindi                             | India                |
| `PT___DDC_CWICR/`       | Portuguese                        | Portugal / Brazil    |
| `RU___DDC_CWICR/`       | Russian                           | Russia               |
| `UK___DDC_CWICR/`       | Ukrainian                         | Ukraine              |
| `US___DDC_CWICR/`       | English                           | United States        |
| `ZH___DDC_CWICR/`       | Chinese                           | China                |

Each directory contains:

- `DDC_CWICR_<REGION>_Catalog.csv` — the resource / work-item
  catalog in CSV form
- `DDC_CWICR_<REGION>_Catalog.xlsx` — the same catalog as Excel
- `*_workitems_costs_resources_DDC_CWICR.parquet` — the full
  work-items dataset in Parquet (columnar, Big-Data friendly)
- `*_workitems_costs_resources_DDC_CWICR_SIMPLE.xlsx` —
  flattened Excel view
- `*_workitems_costs_resources_DDC_CWICR_FORMATTED.xlsx` —
  presentation-ready Excel view
- `*_EMBEDDINGS_3072_DDC_CWICR.snapshot` — Qdrant vector-index
  snapshot (OpenAI `text-embedding-3-large`, 3072-dim)
- `README.md` / `README_*.pdf` / `README_*.txt` — track-local
  documentation

## Resource catalog (`DDC_CWICR_<REGION>_Catalog.csv`)

One row per resource (labour role, material, equipment).

| Column                   | Type    | Description                                                                         |
|--------------------------|---------|-------------------------------------------------------------------------------------|
| `resource_code`          | string  | Stable identifier, track-scoped (e.g. `PU_MEKAKA_KAPUKA`)                           |
| `name`                   | string  | Localised resource name                                                             |
| `type`                   | string  | Resource type (e.g. `Operator`, `Material`, `Equipment`)                            |
| `category`               | string  | Top-level taxonomy bucket                                                           |
| `unit`                   | string  | Measurement unit (e.g. `Machine hours`, `m2`, `m3`, `pcs`, `kg`)                    |
| `price_avg`              | float   | Mean unit price                                                                     |
| `price_min`              | float   | Minimum observed unit price                                                         |
| `price_max`              | float   | Maximum observed unit price                                                         |
| `price_median`           | float   | Median unit price                                                                   |
| `price_variants`         | integer | Number of observed price points used to derive the statistics                        |
| `currency`               | string  | ISO 4217 currency code of the price columns (track-dependent; e.g. `CAD`, `EUR`)    |
| `avg_cost_per_use`       | float   | Average cost contribution of this resource across all work items that consume it     |
| `avg_qty_per_use`        | float   | Average quantity of this resource consumed per work item that consumes it           |
| `usage_count`            | integer | Number of work items that consume this resource                                     |
| `used_in_work_items`     | integer | Count of distinct work items in which the resource appears                           |
| `parent_category`        | string  | Next-up taxonomy level above `category` (e.g. `CONSTRUCTION WORK`)                  |
| `parent_collection`      | string  | Taxonomy collection grouping                                                        |
| `parent_department`      | string  | Higher-level departmental / discipline grouping                                     |
| `parent_section`         | string  | Narrative section / sub-discipline                                                  |

## Work-items dataset (Parquet / Excel)

The `*_workitems_costs_resources_DDC_CWICR.parquet` files are
the primary cost-estimation workload. Each row is a single work
item with the full decomposition across labour, material, and
equipment resources and the associated unit rate, duration, and
output metrics.

The complete schema comprises 85 columns grouped into:

- **Classification hierarchy** — section, department, collection,
  category, sub-category (harmonised across tracks for
  cross-regional comparison).
- **Identifiers** — stable `work_item_code`, localised title,
  description.
- **Units and norms** — measurement unit, output quantity,
  standard duration (labour hours).
- **Resource decomposition** — per-resource `resource_code`,
  consumption quantity, unit price, per-item cost.
- **Cost aggregates** — labour subtotal, material subtotal,
  equipment subtotal, total unit cost.
- **Regional markers** — reference region, currency, last-price
  refresh date.

When using the Parquet file, inspect the schema with
`pyarrow.parquet.read_schema(<file>)` to obtain the authoritative
column-level types.

## Qdrant vector snapshot (`*.snapshot`)

Vector index built from concatenated `title + description +
category + parent_category` fields, encoded with OpenAI
`text-embedding-3-large` (3072-dim, cosine distance). Restore
into a running Qdrant instance with:

```bash
# From the language directory that contains the snapshot
qdrant-client snapshot upload \
  --collection ddc_cwicr_<track> \
  --snapshot <file>.snapshot
```

Derived embeddings are considered derivative works of the
underlying DATA and inherit its CC BY 4.0 licence (see NOTICE —
"AI and vector search notes").

## Source attribution

The dataset was compiled, translated, and harmonised by Licensor
from publicly available construction cost standards and from
expert compilations. The compilation, cross-language mapping, and
taxonomy alignment constitute original editorial work protected
under the EU sui generis database right (Directive 96/9/EC;
§§ 87a-87e UrhG).

Classification codes referenced in the taxonomy columns (DIN 276
cost groups, CSI MasterFormat, UniFormat, OmniClass, NRM element
codes, and regional equivalents ENIR / GESN / FER / NRR / ESN /
AzDTN / ShNQK / MKS ChT / SNT / BNbD / Dinh Muc / Ding'e) are
used as factual references. No standard's copyrighted descriptive
text has been copied into the dataset. All trademarks remain
with their respective holders (see NOTICE).

## Currency and conversion

Prices are delivered in the reference-region currency of each
track (`currency` column). Conversion to other currencies is the
responsibility of the downstream user; no exchange-rate table is
distributed with the dataset because FX rates drift faster than
the dataset refresh cycle.

## Versioning

The dataset versioning is aligned with the GitHub release tag.
Breaking schema changes bump the major version; backwards-
compatible column additions bump the minor version.

## Known limitations

- Regional granularity is at reference-region level, not
  postcode / ZIP.
- Labour costs reflect averaged labour-market compositions at
  the reference region; specific collective agreements may
  differ.
- Material prices lag the live spot market by up to six months;
  commodity-driven line items (steel, concrete, copper, timber)
  should be adjusted against a live index when used in a binding
  estimate.
- Equipment costs assume standard contractor ownership models;
  rental-heavy markets may need adjustment.

---

Dataset questions, schema bugs, and commercial-data enquiries:
**info@datadrivenconstruction.io**
