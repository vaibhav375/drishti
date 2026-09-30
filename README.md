# DRISHTI — deploy branch (Render)

This branch is a **generated bundle** of the DRISHTI web console for the
Render free plan. Don't edit it by hand — source of truth is `main`;
regenerate with `python deploy/build_space.py --target render`.
Full project documentation: [ABOUT.md](ABOUT.md).

DRISHTI is an offline Space Domain Awareness console: it characterizes
on-orbit **behavior** from public element sets (maneuvers, pattern-of-life,
RPO, fragmentation, decay, disposal compliance), scores threat explainably,
and — on a full install — writes LLM reports whose every number is verified.
**Decision-support, not targeting.** Ownership factors are nation-neutral.

What this hosted demo differs in:

- **Public Celestrak data only.** Space-Track history is stripped at build
  time (its user agreement forbids redistribution), so detectors needing a
  long element-set history often show "insufficient data" — the detectors
  correctly refusing to guess.
- **No local LLM** (512 MB host). The report page shows the grounded facts
  the model would narrate; natural-language "ask" falls back to name search.
- **Free-plan sleep:** the service spins down when idle; the first request
  after that takes ~1 min while it wakes and warms its caches.
