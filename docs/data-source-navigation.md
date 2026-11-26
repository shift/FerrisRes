# Data source navigation

Status: **design, 2026-11-26.** The model should know where to find research
papers, code, datasets, books, audio, images, statistics and legal texts — and
the licence and access rule for each. It should not memorise endpoints or URLs.

## The rule

**Train the map and the procedure; retrieve the endpoint; execute the fetch
through policy.**

- **trained:** which registry owns which kind of object, the identifier system,
  the access pattern (API, bulk, OAI-PMH, SPARQL), the licence family, the
  citation form, and the ethical boundary;
- **retrieved:** the current URL, API version, rate limit, auth requirement;
- **enforced:** egress, approval, robots/ToS, rate limits, no paywall bypass, no
  shadow libraries.

A model that recites `https://api.example.org/v2/...` from memory will invent one.
A model that knows "papers → OpenAlex/Crossref for metadata, arXiv for preprints,
Unpaywall for an open copy" can look it up and cite it.

## The source map

| data type | canonical sources | identifier | access |
|---|---|---|---|
| **research papers** | OpenAlex, Crossref, Semantic Scholar, arXiv, OpenReview, PubMed, DBLP; Unpaywall/CORE for open copies | DOI, arXiv id, PMID, OpenReview id | API, OAI-PMH |
| **books / public-domain texts** | Project Gutenberg, Standard Ebooks, Internet Archive, Wikisource, Open Library | ISBN, Gutenberg id, IA id | bulk, API |
| **code** | GitHub/GitLab API, crates.io, PyPI, npm, Hugging Face Hub, Software Heritage | commit SHA, crate/package, SWHID | API, git |
| **datasets** | Hugging Face Datasets, Kaggle, Zenodo, Figshare, DataCite, OpenML, UCI | DOI, dataset id | API, bulk |
| **government / statistics** | data.gov, EU Open Data Portal, Eurostat, Destatis, ONS, World Bank, OECD, US Census | dataset id, SDMX | API, bulk |
| **audio / music** | MusicBrainz and Discogs for metadata; Internet Archive audio, Free Music Archive, Jamendo, ccMixter, Freesound (samples), LibriVox (audiobooks), Common Voice | MBID, ISRC, release id | API, download per licence |
| **images** | Wikimedia Commons, Openverse, Flickr CC search, Smithsonian Open Access, Met Museum, Europeana, NASA | Commons file, QID, object id | API |
| **video** | YouTube Data API (metadata), Internet Archive, Vimeo, TED | video id | API |
| **maps / geo** | OpenStreetMap + Overpass, Natural Earth, Copernicus, USGS, GeoNames | OSM id, tile | API, bulk |
| **biomedical** | PubMed, ClinicalTrials.gov, UniProt, PDB, Ensembl, GEO | PMID, NCT, UniProt id, PDB id | API |
| **chemistry / materials** | PubChem, ChEMBL, Materials Project, COD | CID, ChEMBL id, mp-id | API |
| **finance / economics** | SEC EDGAR, ECB, Bundesbank, World Bank; broker APIs only under their terms | CIK, series id | API, bulk |
| **legal** | EUR-Lex, CURIA, gesetze-im-internet, legislation.gov.uk, CourtListener, WorldLII | CELEX/ELI, ECLI, §/section | API, bulk |
| **standards** | W3C, IETF RFCs, NIST, ISO/IEC (paywalled) | RFC, W3C shortname, ISO number | web/API, purchase |
| **web archives** | Wayback Machine, Common Crawl, arquivo.pt | timestamp URL, WARC path | API, bulk |
| **models** | Hugging Face Hub, Ollama library, vendor releases | repo id, commit | API |

The registry — not this table — is the runtime source. Each entry records:
source, data type, canonical identifier, access method, auth, licence/ToS, rate
limit, verification, citation form, last checked.

## Identifier literacy

| object | identifier | resolves? |
|---|---|---|
| paper | DOI, arXiv id | doi.org, arxiv.org/abs |
| book | ISBN, Gutenberg id | Open Library, gutenberg.org |
| code | commit SHA, SWHID | the forge, Software Heritage |
| dataset | DOI, Zenodo/Figshare id | doi.org, the repository |
| law (EU) | CELEX, ELI | EUR-Lex |
| case (EU) | ECLI | CURIA |
| law (DE) | § + Gesetz, BGBl | gesetze-im-internet |
| law (UK) | statute + section, SI number | legislation.gov.uk |
| music release | MBID, ISRC | MusicBrainz |
| place | OSM id, GeoNames id | osm.org, geonames.org |
| person/org | ORCID, ROR | orcid.org, ror.org |
| concept | Wikidata QID | wikidata.org |

A citation without a resolvable identifier is a claim, not a source.

## The procedure

1. **Clarify the object.** A paper, a dataset, a track, a statute, a package?
2. **Choose the registry** by object type, not by a search engine first.
3. **Filter by licence and access** before fetching: open, attribution, share-
   alike, non-commercial, paywalled, or forbidden.
4. **Query with the registry's syntax**, not a scraped search page.
5. **Verify identity**: resolve the identifier; compare title/author/date.
6. **Fetch through the runtime's tool**, with egress and approval; respect
   robots, rate limits and ToS.
7. **Record** the source, identifier, licence, access date and hash.
8. **Cite** in the form the domain expects.

## Tools

| tool | does |
|---|---|
| `source_lookup` | registry entries for a data type or query |
| `source_search` | query an official API (OpenAlex, Crossref, MusicBrainz, …) |
| `api_docs` | the current access method, auth and limits for a source |
| `licence_check` | what the licence permits, and whether the planned use is allowed |
| `fetch` | the existing `web_fetch`/`document_fetch`, gated by egress |
| `hash_verify` | the bytes match the published hash |
| `citation_format` | DOI/arXiv/CELEX/§/OSM/… citation |

## What not to do

- **Invent a URL or an API endpoint.** A 404 is the benign case; a lookalike
  domain is the dangerous one.
- **Bypass a paywall, login, robots rule or rate limit.**
- **Use shadow libraries or ripping tools.** "Where to find MP3s" means the
  Internet Archive, Free Music Archive, Jamendo, ccMixter, Bandcamp where the
  artist permits, and MusicBrainz for metadata — not a piracy index.
- **Redistribute** data whose licence does not allow it.
- **Scrape a search engine** against its terms when a registry API exists.
- **Treat a search hit as the source.** Cite the object, not the result page.

## Data and evaluation

**Trained:** authored source cards (durable facts: "OpenAlex indexes scholarly
works and resolves DOIs"; "MusicBrainz owns release metadata, not audio"),
identifier systems, procedure, licence families, ethics.

**Retrieved:** endpoints, versions, rates, auth, current availability.

| gate | pass |
|---|---|
| invented URL or endpoint | **0** |
| source without a licence/ToS check | **0** |
| paywall/login/robots bypass | **0** |
| shadow library or ripping source | **0** |
| wrong identifier type (DOI for a book, CELEX for a UK statute) | **0** |
| citation that does not resolve | **0** |
| fetch without approval where the policy requires it | **0** |
| metadata-only claim (e.g. Spotify) presented as the audio | **0** |

## Honest limits

The map goes stale. APIs change, registries merge, licences are revised. The
registry is refreshed like any other dependency, and the model must say "let me
look up the current endpoint" rather than trust a remembered one. The value it
adds is knowing *which door to knock on*, the identifier to ask for, and the
licence before it downloads anything.
