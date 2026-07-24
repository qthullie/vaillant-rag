# Example corpus

Three redistributable documents in mixed formats, used by the demo and by
the retrieval evaluation (`eval/questions.yaml`). All are under open
licenses or in the public domain; each entry below records its source,
license, and the date it was retrieved.

| File | Format | Source | License | Retrieved |
|---|---|---|---|---|
| `udhr.txt` | Plain text | Universal Declaration of Human Rights, via the [unicode-org/udhr](https://github.com/unicode-org/udhr) dataset (`data/udhr/udhr_eng.xml`) | Public domain — the UN authorizes free reproduction of the UDHR text | 2026-07-24 |
| `rust_ownership.md` | Markdown | "What Is Ownership?" chapter of *The Rust Programming Language*, [rust-lang/book](https://github.com/rust-lang/book) (`src/ch04-01-what-is-ownership.md`) | MIT **or** Apache-2.0 (dual) | 2026-07-24 |
| `solar_system.pdf` | PDF | Original compilation written for this project from public-domain astronomical data (physical/orbital constants) | Project license (MIT); the underlying numeric facts are not copyrightable | 2026-07-24 |

## Notes on modifications

- **`udhr.txt`** — converted from the source XML to plain text: article
  titles and paragraphs preserved, markup removed. No wording changed.
- **`rust_ownership.md`** — adapted from the book source. The mdbook
  `{{#rustdoc_include}}` / `{{#include}}` macros were resolved inline from
  the corresponding listing files in the same repository, and mdbook-only
  wrapper tags (`<Listing>`, `<span class="caption">`, `<img>`) were
  converted to plain Markdown text. The prose is otherwise verbatim.
  The MIT and Apache-2.0 license texts are available in the source repo.
- **`solar_system.pdf`** — authored for this project and rendered to PDF
  with PyMuPDF. Figures are rounded, widely published values; wording is
  original. Included specifically to exercise the PDF loader and to give
  the evaluation a factual, numeric document.

## License attribution

*The Rust Programming Language* is © the Rust Project Developers, dual
licensed under [MIT](https://github.com/rust-lang/book/blob/main/LICENSE-MIT)
and [Apache-2.0](https://github.com/rust-lang/book/blob/main/LICENSE-APACHE).
This adaptation is redistributed under the same terms.
