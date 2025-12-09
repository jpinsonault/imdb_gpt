from scripts.autoencoder.print_model import summarize_model


def _print_section(title, rows):
    if not rows:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print("[no modules]")
        return

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    header = (
        f"{'Idx':>4}  {'Name':<30}  {'Type':<35}  "
        f"{'Input':<20}  {'Output':<20}  {'Params':>10}  {'Train':>10}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows):
        indent = "  " * r["depth"]
        name = (indent + r["name"]).strip()
        if len(name) > 30:
            name = name[:27] + "..."
        t = r["type"]
        if len(t) > 35:
            t = t[:32] + "..."
        inp = r["in"]
        out = r["out"]
        if len(inp) > 20:
            inp = inp[:17] + "..."
        if len(out) > 20:
            out = out[:17] + "..."
        print(
            f"{i:4d}  {name:<30}  {t:<35}  "
            f"{inp:<20}  {out:<20}  {r['params']:10d}  {r['trainable']:10d}"
        )

    section_total = sum(r["params"] for r in rows)
    section_train = sum(r["trainable"] for r in rows)
    print("-" * len(header))
    print(
        f"{'Section params:':<20} {section_total:d}    "
        f"{'Trainable:':<12} {section_train:d}    "
        f"{'Frozen:':<10} {section_total - section_train:d}"
    )


def print_hybrid_model_sections(model, sample_inputs_cpu, sample_idx, device):
    sample_inputs = {
        "field_tensors": [x.to(device) for x in sample_inputs_cpu],
        "batch_indices": sample_idx,
    }

    summary = summarize_model(
        model,
        sample_inputs,
        max_depth=None,
        print_fn=lambda *args, **kwargs: None,
    )
    rows = summary["layers"]

    def top_name(r):
        name = r["name"]
        if name == "(root)":
            return "(root)"
        return name.split(".")[0]

    movie_field_roots = {"movie_embeddings", "field_decoder"}
    people_field_roots = {"person_embeddings", "person_field_decoder"}
    movie_set_roots = {"heads"}
    people_set_roots = {"movie_heads"}

    movie_field_rows = [r for r in rows if top_name(r) in movie_field_roots]
    people_field_rows = [r for r in rows if top_name(r) in people_field_roots]
    movie_set_rows = [r for r in rows if top_name(r) in movie_set_roots]
    people_set_rows = [r for r in rows if top_name(r) in people_set_roots]

    print("\n" + "=" * 80)
    print("HybridSetModel structure")
    print("=" * 80)

    _print_section("Movie field path (embeddings + decoder)", movie_field_rows)
    _print_section("People field path (embeddings)", people_field_rows)
    _print_section("Set decoding: movie → people (heads)", movie_set_rows)
    _print_section("Set decoding: person → movies (movie_heads)", people_set_rows)

    embed_roots = {"movie_embeddings", "person_embeddings"}
    field_roots = {"field_decoder", "person_field_decoder"}
    set_roots = {"heads", "movie_heads"}

    embed_params = sum(r["params"] for r in rows if top_name(r) in embed_roots)
    field_params = sum(r["params"] for r in rows if top_name(r) in field_roots)
    set_params = sum(r["params"] for r in rows if top_name(r) in set_roots)
    total_params = summary["total_params"]

    other_params = total_params - (embed_params + field_params + set_params)

    print("\n" + "=" * 80)
    print("Parameter summary by component")
    print("=" * 80)
    print(f"{'Embeddings:':<20} {embed_params:d}")
    print(f"{'Field decoders:':<20} {field_params:d}")
    print(f"{'Set heads:':<20} {set_params:d}")
    print(f"{'Other:':<20} {other_params:d}")
    print(f"{'Total:':<20} {total_params:d}")
    print("=" * 80 + "\n")
