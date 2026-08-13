---
title: Postal Codes and Addresses
---

# Postal Codes and Addresses

Stickler compares postal codes **exactly**. `"98101-1234"` does not match
`"98101 1234"`, and `"98101"` does not match `"98101-1234"`.

That is deliberate, and it is probably not what you want for your data. This
page explains why the library refuses to guess, and how to get the behaviour
your country and pipeline need.

## Why there is no `PostalCodeComparator`

A generic normalizer would be right for the United States and quietly wrong
elsewhere:

| country | example | what "strip punctuation and casefold" does |
|---|---|---|
| US | `98101-1234` | correct: ZIP+4 separator is cosmetic |
| UK | `SW1A 1AA` | **wrong**: the space separates outward and inward codes |
| Netherlands | `1012 AB` | **wrong**: the space separates digits from letters |
| Canada | `K1A 0B1` | **wrong**: same structural split as the UK |
| Brazil | `01310-100` | correct: hyphen is cosmetic |

Silently applying US rules to a UK postcode is worse than not matching, because
the result looks like a success. Compare this with
[phone numbers](#contrast-phone-numbers), where the library *does* normalize,
because a small library encodes the whole world's rules.

## What not to reach for

**Do not use a similarity comparator.** Edit distance cannot separate the two
cases you care about:

```python
LevenshteinComparator().compare("98101-1234", "98101 1234")   # 0.9  same code
LevenshteinComparator().compare("98101-1234", "98102-1234")   # 0.9  DIFFERENT code
```

Both score `0.9`. There is no threshold that accepts the first and rejects the
second, so any threshold you pick is arbitrary. The same trap is worse for
phone numbers, where a different number can score *higher* than a reformatted
identical one.

## Getting the behaviour you want

### Option 1: normalize in your own comparator (recommended)

If you know your data's country, the rule is a few lines and you control it:

```python
import re
from stickler import BaseComparator

class USZipComparator(BaseComparator):
    """US ZIP codes, ignoring the ZIP+4 separator.

    Treats "98101-1234" and "98101 1234" as equal. Does NOT treat "98101" as
    equal to "98101-1234" -- those are different levels of precision, and
    collapsing them hides a real difference in what was extracted.
    """

    def _compare(self, gt, pred) -> float:
        return 1.0 if self._canonical(gt) == self._canonical(pred) else 0.0

    @staticmethod
    def _canonical(value) -> str:
        digits = re.sub(r"\D", "", str(value))
        # 5-digit and 9-digit (ZIP+4) forms stay distinct.
        return digits


class UKPostcodeComparator(BaseComparator):
    """UK postcodes, normalizing spacing and case but keeping structure.

    "sw1a1aa", "SW1A 1AA" and "SW1A  1AA" are the same postcode. The outward
    and inward parts are re-joined canonically rather than concatenated, so
    the structural split survives.
    """

    def _compare(self, gt, pred) -> float:
        return 1.0 if self._canonical(gt) == self._canonical(pred) else 0.0

    @staticmethod
    def _canonical(value) -> str:
        compact = re.sub(r"\s+", "", str(value)).upper()
        # Inward code is always the last three characters.
        return f"{compact[:-3]} {compact[-3:]}" if len(compact) > 3 else compact
```

Then attach it to the field:

```python
class Address(StructuredModel):
    zip_code: str = ComparableField(comparator=USZipComparator(), threshold=1.0)
```

Both examples implement `_compare`, not `compare` -- see
[Comparators](README.md#none-handling) for why. `None` handling is inherited.

### Option 2: use a third-party library

If you need real address intelligence rather than string normalization, these
exist. None is bundled, because each carries a cost stickler will not impose on
every install:

| library | scope | cost |
|---|---|---|
| [`libpostal`](https://github.com/openvenues/libpostal) (via `postal`) | international, best in class | a C library plus ~2 GB of data |
| [`zipcodes`](https://pypi.org/project/zipcodes/) | US lookup and validation | 1.9 MB, no dependencies |
| [`pgeocode`](https://pypi.org/project/pgeocode/) | postal code geocoding | pulls `numpy` and `pandas` |
| [`usaddress`](https://pypi.org/project/usaddress/) | US full-address parsing | 8 dependencies, CRF model |

`zipcodes` is the practical choice for a US-only pipeline. `zipcodes.matching()`
resolves both `"98101"` and `"98101-1234"` to `98101`, which is a clean
canonical form to compare on:

```python
import zipcodes
from stickler import BaseComparator

class ZipLookupComparator(BaseComparator):
    """Compare US ZIPs by the 5-digit code they resolve to."""

    def _compare(self, gt, pred) -> float:
        return 1.0 if self._resolve(gt) == self._resolve(pred) else 0.0

    @staticmethod
    def _resolve(value):
        matches = zipcodes.matching(str(value))
        return matches[0]["zip_code"] if matches else None
```

Note the consequence: this makes `"98101"` and `"98101-1234"` equal. That may
be what you want, or it may hide that your extractor dropped the +4.

### Option 3: normalize before evaluating

If every consumer of your data wants the same canonical form, do it upstream and
leave stickler exact. Evaluation then measures extraction quality rather than
formatting, and the canonical form is one you can inspect.

## Deciding what "equal" means

Worth settling before you pick an implementation, because the library cannot
decide it for you:

- Is `98101` the same as `98101-1234`? Same place, different precision. If your
  extractor is *supposed* to capture the +4, treating them as equal hides a
  miss.
- Is casing significant? For UK and Canadian codes, no. For codes containing a
  meaningful letter case, possibly.
- Do you care about validity, or only agreement? A comparator answers "did the
  extractor agree with ground truth". A validator answers "is this a real
  postal code". They are different questions.

## Contrast: phone numbers

Stickler *does* normalize phone numbers, via
[`PhoneComparator`](README.md), because
[libphonenumber](https://github.com/google/libphonenumber) encodes every
country's rules in 450 KB with no dependencies. `"206-555-0100"`,
`"(206) 555-0100"` and `"+1-206-555-0100"` all compare equal.

There is no equivalent for postal codes at that weight, which is the whole
reason this page exists rather than a comparator.
