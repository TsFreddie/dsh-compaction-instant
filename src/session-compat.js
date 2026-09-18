/**
 * Cross-version access to one session's durable event log.
 *
 * Older harness releases exposed the append-only log as the plain array
 * `session.events`. Current releases keep the log private and publish two
 * accessors instead: `eventAt(seq)` for one exact event and
 * `snapshotEvents(from, to)` for a frozen snapshot of a sequence range.
 *
 * Both helpers prefer the accessor pair when it is present and fall back to
 * the legacy array, so the compiler, the recall tools, and the search tool
 * work unchanged on either harness generation.
 * @module dsh-compaction-instant/session-compat
 */

/**
 * Return a frozen snapshot of every event in the durable log.
 * @param session - session-shaped value with `snapshotEvents()` or legacy `events`.
 * @returns the frozen event array, or an empty array when neither shape exists.
 */
export function sessionEvents(session) {
  if (typeof session?.snapshotEvents === "function") return session.snapshotEvents();
  return session?.events ?? [];
}

/**
 * Return the immutable event stored at one exact sequence number.
 * @param session - session-shaped value with `eventAt()` or legacy `events`.
 * @param seq - event sequence number.
 * @returns the accepted event, or undefined when the log does not contain it.
 */
export function sessionEventAt(session, seq) {
  if (typeof session?.eventAt === "function") return session.eventAt(seq);
  return session?.events?.[seq];
}
