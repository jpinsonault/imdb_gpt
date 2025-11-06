# scripts/autoencoder/share_policy.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type

import types

from .fields import (
    BaseField,
    TextField,
    NumericDigitCategoryField,
    MultiCategoryField,
    SingleCategoryField,
    ScalarField,
    BooleanField,
)
from ..autoencoder.row_autoencoder import RowAutoencoder


@dataclass
class _Group:
    space_id: str
    field_cls: Type[BaseField]
    include: Optional[Sequence[Tuple[RowAutoencoder, str]]] = None


class SharePolicy:
    def __init__(self):
        self._groups: List[_Group] = []

    def group(
        self,
        space_id: str,
        field_cls: Type[BaseField],
        include: Optional[Sequence[Tuple[RowAutoencoder, str]]] = None,
    ) -> "SharePolicy":
        self._groups.append(_Group(space_id=space_id, field_cls=field_cls, include=include))
        return self

    def plan(self, *aes: RowAutoencoder) -> dict:
        out = {}
        for g in self._groups:
            fields = self._resolve_fields(g, aes)
            out[g.space_id] = [f"{type(f).__name__}:{f.name}" for f in fields]
        return out

    def apply(self, *aes: RowAutoencoder) -> None:
        for g in self._groups:
            fields = self._resolve_fields(g, aes)
            if not fields:
                continue
            self._merge_stats_for_group(g.field_cls, fields)
            self._tie_weights(g.space_id, fields)

    def _resolve_fields(self, grp: _Group, aes: Iterable[RowAutoencoder]) -> List[BaseField]:
        if grp.include:
            picked: List[BaseField] = []
            for ae, fname in grp.include:
                f = self._field_by_name(ae, fname)
                if isinstance(f, grp.field_cls):
                    picked.append(f)
            return picked
        picked = []
        for ae in aes:
            for f in ae.fields:
                if isinstance(f, grp.field_cls):
                    picked.append(f)
        return picked

    def _field_by_name(self, ae: RowAutoencoder, name: str) -> BaseField:
        for f in ae.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def _merge_stats_for_group(self, field_cls: Type[BaseField], fields: List[BaseField]) -> None:
        if not fields:
            return
        if issubclass(field_cls, TextField):
            self._merge_text(fields)  # type: ignore[arg-type]
            return
        if issubclass(field_cls, NumericDigitCategoryField):
            self._merge_numeric_digits(fields)  # type: ignore[arg-type]
            return
        if issubclass(field_cls, MultiCategoryField):
            self._merge_multi_cat(fields)  # type: ignore[arg-type]
            return
        if issubclass(field_cls, SingleCategoryField):
            self._merge_single_cat(fields)  # type: ignore[arg-type]
            return
        if issubclass(field_cls, ScalarField):
            self._merge_scalar(fields)  # type: ignore[arg-type]
            return
        if issubclass(field_cls, BooleanField):
            self._merge_boolean(fields)  # type: ignore[arg-type]
            return

    def _tie_weights(self, space_id: str, fields: List[BaseField]) -> None:
        if not fields:
            return

        enc_ref: dict[str, Any] = {"m": None}
        dec_ref: dict[str, Any] = {"m": None}

        enc_template = fields[0].build_encoder
        dec_template = fields[0].build_decoder

        def enc_wrapper(self, latent_dim: int, _tmpl=enc_template, _ref=enc_ref):
            if _ref["m"] is None:
                _ref["m"] = _tmpl(latent_dim)
            return _ref["m"]

        def dec_wrapper(self, latent_dim: int, _tmpl=dec_template, _ref=dec_ref):
            if _ref["m"] is None:
                _ref["m"] = _tmpl(latent_dim)
            return _ref["m"]

        for f in fields:
            f.build_encoder = types.MethodType(enc_wrapper, f)
            f.build_decoder = types.MethodType(dec_wrapper, f)

    def _merge_text(self, fields: List[TextField]) -> None:
        all_texts: List[str] = []
        for f in fields:
            all_texts.extend(getattr(f, "texts", []) or [])
        max_user = None
        has_any_user = any(getattr(f, "user_max_length", None) is not None for f in fields)
        if has_any_user:
            max_user = max(int(getattr(f, "user_max_length", 0) or 0) for f in fields)
        for f in fields:
            f.texts = list(all_texts)
            f.tokenizer = None
            f.dynamic_max_len = 0
            f.max_length = None
            f.pad_token_id = None
            f.null_token_id = None
            if has_any_user:
                f.user_max_length = int(max_user)
            else:
                f.user_max_length = None

    def _merge_numeric_digits(self, fields: List[NumericDigitCategoryField]) -> None:
        all_vals: List[float] = []
        any_nan = False
        any_neg = False
        max_frac = 0
        for f in fields:
            any_nan = any_nan or bool(getattr(f, "has_nan", False))
            any_neg = any_neg or bool(getattr(f, "has_negative", False))
            max_frac = max(max_frac, int(getattr(f, "fraction_digits", 0)))
            all_vals.extend(list(getattr(f, "data_points", []) or []))
        for f in fields:
            f.has_nan = bool(any_nan)
            f.has_negative = bool(any_neg)
            f.fraction_digits = int(max_frac)
            f.data_points = list(all_vals)
            f.integer_digits = None
            f.total_positions = None

    def _merge_multi_cat(self, fields: List[MultiCategoryField]) -> None:
        cats = set()
        counts: dict[str, int] = {}
        for f in fields:
            for c, n in (getattr(f, "category_counts", {}) or {}).items():
                cats.add(c)
                counts[c] = counts.get(c, 0) + int(n)
            for c in (getattr(f, "category_list", []) or []):
                cats.add(c)
        cat_list = sorted(cats)
        for f in fields:
            f.category_set = set(cat_list)
            f.category_list = list(cat_list)
            f.category_counts = dict(counts)

    def _merge_single_cat(self, fields: List[SingleCategoryField]) -> None:
        cats = set()
        counts: dict[str, int] = {}
        for f in fields:
            for c, n in (getattr(f, "category_counts", {}) or {}).items():
                cats.add(c)
                counts[c] = counts.get(c, 0) + int(n)
            for c in (getattr(f, "category_list", []) or []):
                cats.add(c)
        cat_list = sorted(cats)
        for f in fields:
            f.category_set = set(cat_list)
            f.category_list = list(cat_list)
            f.category_counts = dict(counts)

    def _merge_scalar(self, fields: List[ScalarField]) -> None:
        n = 0
        s = 0.0
        ss = 0.0
        min_v = float("inf")
        max_v = float("-inf")
        for f in fields:
            n += int(getattr(f, "n", 0))
            s += float(getattr(f, "sum_", 0.0))
            ss += float(getattr(f, "sum_sq", 0.0))
            min_v = min(min_v, float(getattr(f, "min_val", float("inf"))))
            max_v = max(max_v, float(getattr(f, "max_val", float("-inf"))))
        if min_v == float("inf"):
            min_v = 0.0
        if max_v == float("-inf"):
            max_v = 0.0
        for f in fields:
            f.n = int(n)
            f.sum_ = float(s)
            f.sum_sq = float(ss)
            f.min_val = float(min_v)
            f.max_val = float(max_v)
            f.mean_val = 0.0
            f.std_val = 1.0

    def _merge_boolean(self, fields: List[BooleanField]) -> None:
        total = 0
        ones = 0
        for f in fields:
            total += int(getattr(f, "count_total", 0))
            ones += int(getattr(f, "count_ones", 0))
        for f in fields:
            f.count_total = int(total)
            f.count_ones = int(ones)
