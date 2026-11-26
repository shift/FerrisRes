# Data retention and privacy

Status: **policy defaults, 2026-11-26.** These are the defaults the code should
enforce; legal review is required before a public deployment.

## Data classes

| class | examples | default retention | where |
|---|---|---|---|
| **chat** | messages, attachments | configurable, default 90 days; deletable | session store |
| **soul** | the committed identity text | until replaced or deleted | entity store |
| **profile** | facts about the principal | until deleted | entity store |
| **memory** | entities, spans, evidence | revision-bound; kept until deleted | entity store |
| **run records** | hashes, counts, device, outcome | kept for audit | files/entities |
| **raw logs and transcripts** | prompt/response text | **not published**, short local retention | local only |
| **training corpora** | licensed shards | per licence | corpus store |
| **eval sets** | held-out data | kept, never trained | eval store |
| **secrets** | keys, tokens | **never persisted** | environment only |

## Rules

1. **Local first.** The default deployment stores data on the operator's machine;
   a hosted provider sees a request only when the operator configures it.
2. **No private data in training by default.** Souls, chat, run records and child
   data are not corpus rows. An opt-in requires consent, redaction and a recorded
   decision; a generated row that contains any of them is rejected.
3. **Deletion cannot unlearn.** A deletion removes rows, indexes and embeddings;
   it cannot remove a fact a model was trained on. Say that to the user; do not
   imply otherwise.
4. **Secrets never leave the process.** They are read from the environment, not
   written to config, not logged, not sent to a hosted generator.
5. **Child data is minimised.** The child principal's chat is read-only, written
   only through the declared skill author, and excluded from any corpus.
6. **Raw logs are private.** They are not committed, uploaded, pasted into a
   report or used as a demo (`docs/dataset-publication-policy.md` in the harness).

## Deletion

A principal deletion must remove:

- messages and attachments for that principal;
- entities, spans and embeddings scoped to its project;
- derived index entries and caches keyed by it;
- scheduled work and pending approvals.

It must record the deletion as an audit event (id, time, counts), not the content.
Backups expire on their own schedule; the deletion record names the last backup
that may still contain the data.

A test asserts that a deletion removes the rows **and** the index entries; a
deletion that only hides a row is a bug.

## Hosted generators

- Only models on the allowlist (`model-licensing.md`) may receive a request.
- The prompt is authored or synthetic; a soul, chat message, transcript,
  credential or customer document is never sent.
- A rate or auth failure stops generation; it will not succeed on retry.
- The provider's terms snapshot is recorded per generated row.

## Access and export

- The principal can export their souls, profile and memory as a portable file.
- An operator can audit who accessed what from the run records; the records hold
  ids and counts, not message text.
- Data residency follows the deployment: local by default; a hosted deployment
  documents its region.

## Open questions for legal review

- Retention defaults per jurisdiction (the 90-day chat default is a policy
  choice, not a legal conclusion).
- Whether a hosted generator's jurisdiction changes the transfer analysis.
- The exact wording of the deletion notice ("removed from our systems; may
  persist in backups until …").
