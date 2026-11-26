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
| **standards** | IETF RFCs/BCP/STD, W3C TR, WHATWG, ISO/IEC, IEEE, ECMA, ITU-T, NIST, IANA, OASIS, Unicode, 3GPP/ETSI | RFC number, W3C shortname + date, ISO/IEC number:year, IEEE number-year, ECMA number edition, NIST SP/FIPS rev., registry+value | web/API, some purchase |
| **web archives** | Wayback Machine, Common Crawl, arquivo.pt | timestamp URL, WARC path | API, bulk |
| **models** | Hugging Face Hub, Ollama library, vendor releases | repo id, commit | API |

The registry — not this table — is the runtime source. Each entry records:
source, data type, canonical identifier, access method, auth, licence/ToS, rate
limit, verification, citation form, last checked.

## Standards bodies and versioning

A standard is not a web page. The failure mode is a citation that is real,
plausible and **the wrong version** — an obsoleted RFC, a W3C Working Draft
quoted as a Recommendation, or an ISO clause copied from a paywalled PDF.

| body | where | versioning rule | access |
|---|---|---|---|
| **IETF** | `rfc-editor.org/rfc/rfcNNNN`, `datatracker.ietf.org` | an RFC number is **immutable**; `obsoletes`/`updates` and errata say what changed; drafts expire and are not normative | free |
| **IETF BCP/STD** | same, subseries numbers | a BCP number can point at a newer RFC set | free |
| **W3C** | `w3.org/TR/<shortname>` | dated REC/CR/PR/WD vs an Editor's Draft vs a **Living Standard**; cite the dated version and check errata | free |
| **WHATWG** | `whatwg.org` | living standard only; cite the commit/date, not a "version" | free |
| **ISO/IEC** | iso.org, national bodies (DIN, BSI, ANSI) | edition + year + amendments/corrigenda; national adoption may differ | **paywalled** |
| **IEEE** | ieee.org (802.x, 1003.1) | number + year; amendments are separate documents | **paywalled** |
| **ECMA** | ecma-international.org (ECMA-262) | edition + year; TC39 proposals are stages, not standards | free |
| **ITU-T** | itu.int (X.509, …) | Recommendation number + version/date | free |
| **NIST** | csrc.nist.gov (SP 800-x, FIPS) | revision number is part of the citation (`SP 800-53 Rev. 5`) | free |
| **IANA** | iana.org/registries | registry + value; the registry is the authority, not a blog | free |
| **OASIS** | oasis-open.org (SAML, ODF) | version + errata | free |
| **Unicode** | unicode.org (UAX/UTS) | version (`Unicode 16.0`, `UAX #15`) | free |
| **3GPP / ETSI** | 3gpp.org, etsi.org | release + spec number; ETSI publishes adopted versions | free/registration |

**Rules for a standards answer**

1. Name the body and the series, then the exact document.
2. Say whether it is immutable (RFC), dated (W3C REC), living (WHATWG), or an
   edition with amendments (ISO/IEC).
3. Check `obsoleted-by` / `updates` / errata before calling it current.
4. Cite the section from the document's own table of contents; never invent a
   clause number.
5. For a paywalled standard, cite the standard and its scope/abstract; do not
   reproduce the clause.
6. Never present a draft, a stage proposal or a blog summary as the standard.

### EU-first: CEN, CENELEC, ETSI, and the OJ citation

We are EU-based, so the European standardisation system comes first. The three
ESOs are **CEN**, **CENELEC** and **ETSI**. A standard is **harmonised** only
when its reference is **cited in the Official Journal (OJEU)** under a directive
or regulation; that citation is what gives presumption of conformity. A document
that looks official but has no OJ citation is not harmonised.

| act / area | standards landscape | note |
|---|---|---|
| **AI Act** (EU) 2024/1689 | CEN-CENELEC **JTC 21**; harmonised standards and a GPAI code of practice | check the OJ; standards may be pending |
| **Cyber Resilience Act** (EU) 2024/2847 | CEN-CENELEC **JTC 13** work programme | obligations phase in; verify dates |
| **NIS2** (EU) 2022/2555 | EN ISO/IEC 27001 (via CEN), ETSI, national schemes | national implementation differs |
| **eIDAS 2.0** (EU) 2024/1183 | ETSI EN 319 4xx; EUDI Wallet ARF | ARF is architecture, not a standard |
| **RED cybersecurity** 2014/53/EU + Del. Reg. 2022/30 | **EN 18031-1/-2/-3** | confirm the OJ citation and date |
| **Accessibility** (EAA 2019/882, Web Accessibility Directive) | **EN 301 549**, WCAG | ETSI publishes EN 301 549 |
| **EMC / LVD / Machinery / RoHS / REACH** | 2014/30/EU, 2014/35/EU, 2023/1230, 2011/65/EU, REACH | harmonised standards per act |
| **GDPR** | no harmonised standard; EDPB guidance; EN ISO/IEC 27701 is a management extension, not law | do not present a certificate as compliance |
| **DORA** (EU) 2022/2554 | ICT risk and incident standards | financial sector |

**EU sources and access**

| source | what | access |
|---|---|---|
| EUR-Lex / OJEU | the act, the ELI/CELEX id, the **harmonised-standard citation** | free |
| CEN / CENELEC catalogue | EN references, scopes, national adoptions | **paywalled**; national members (DIN, NEN, AFNOR, UNI) sell the text |
| ETSI standards search | EN/TS/GS, versions (e.g. `EN 303 645 V2.1.1`) | **free** |
| ENISA / EU Cybersecurity Certification | EUCC, EUCS, guidance | free |
| data.europa.eu / Eurostat / EU Vocabularies | EU data, statistics, controlled vocabularies | free |
| national adoption | `DIN EN`, `NEN-EN`, `NF EN`, `UNI EN` | paywalled; same EN, national cover |

**Rules for an EU standards answer**

1. Name the **act** (CELEX/ELI) and the **standard** (EN number + year), and say
   whether the OJ cites it.
2. **Harmonised ≠ published.** No OJ citation, no presumption of conformity.
3. A **withdrawn** OJ citation ends the presumption; check the current OJ list.
4. A **national adoption** (`DIN EN …`) is the same EN with a national cover;
   cite the EN and note the adoption.
5. CEN/CENELEC texts are paywalled: cite the standard and its scope, do not
   reproduce clauses. ETSI texts are free: cite the version.
6. A standard is **not law**. The act binds; the standard is one route to
   conformity. Do not say "the AI Act requires EN …" unless the OJ says so.
7. For a German implementation, the **national act** (e.g. the NIS2
   Umsetzungsgesetz) and the **BSI** guidance sit beside the EU act.

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
| standard (IETF) | RFC number, BCP/STD | rfc-editor.org, datatracker |
| standard (W3C) | shortname + dated version | w3.org/TR |
| standard (ISO/IEC) | number:year (+ amendment) | iso.org / national body |
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
| obsoleted RFC cited as current without a note | **0** |
| living standard cited as a dated Recommendation (or the reverse) | **0** |
| invented clause or section number | **0** |
| paywalled standard reproduced beyond its scope/abstract | **0** |
| EN cited as harmonised without an OJ citation | **0** |
| a withdrawn OJ citation presented as current | **0** |
| national adoption confused with the EN, or the EN with the act | **0** |
| a standard presented as law ("the AI Act requires EN …" without the OJ) | **0** |

## Honest limits

The map goes stale. APIs change, registries merge, licences are revised. The
registry is refreshed like any other dependency, and the model must say "let me
look up the current endpoint" rather than trust a remembered one. The value it
adds is knowing *which door to knock on*, the identifier to ask for, and the
licence before it downloads anything.
