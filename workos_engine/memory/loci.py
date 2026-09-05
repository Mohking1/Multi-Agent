class SpatialLociManager:
    """
    Manages spatial Loci memory organization (Wings & Halls).
    Provides normalization, domain routing, and compact index formatting.
    """

    CANONICAL_WINGS = {
        "people": ["contacts", "roles", "preferences", "teams", "general"],
        "projects": ["decisions", "architecture", "tasks", "roadmaps", "general"],
        "workflows": ["preferences", "runs", "templates", "policies", "general"],
        "knowledge": ["python", "databases", "tools", "policies", "general"],
        "general": ["general"],
    }

    def normalize_locus(self, wing: str | None = None, hall: str | None = None) -> tuple[str, str]:
        """
        Normalizes wing and hall names into clean, lowercase tokens.
        Defaults to ('general', 'general') if unspecified.
        """
        norm_wing = (wing or "general").strip().lower().replace(" ", "_")
        norm_hall = (hall or "general").strip().lower().replace(" ", "_")
        return norm_wing or "general", norm_hall or "general"

    # Table-driven topic routing rules (keywords_set, wing, hall)
    ROUTING_RULES: list[tuple[tuple[str, ...], str, str]] = [
        (("person", "who is", "contact", "email", "phone", "colleague", "role", "manager", "accountant"), "people", "contacts"),
        (("prefer", "format", "style", "always", "never", "summarize as", "bullet points", "paragraphs"), "workflows", "preferences"),
        (("architecture", "decision", "database", "framework", "refactor", "roadmap", "task"), "projects", "decisions"),
        (("python", "sqlite", "elasticsearch", "docling", "api", "protocol", "fact"), "knowledge", "general"),
    ]

    def route_topic(
        self,
        topic: str,
        default_wing: str = "general",
        default_hall: str = "general",
    ) -> tuple[str, str]:
        """
        Maps a topic to a spatial locus (wing, hall) using declarative routing rules.
        """
        topic_lower = topic.lower()
        for keywords, wing, hall in self.ROUTING_RULES:
            if any(k in topic_lower for k in keywords):
                return wing, hall

        return default_wing, default_hall

    def format_summary(
        self,
        loci_data: dict[str, dict[str, list[str]]],
        max_items_per_hall: int = 6,
    ) -> str:
        """
        Generates a token-efficient spatial Loci memory index (< 150-200 tokens).
        Format:
        [Spatial Loci Memory Index]
        - people/contacts (2 keys): arvind_role, sarah_email
        - projects/decisions (1 key): sqlite_db
        - workflows/preferences (1 key): summary_format
        """
        if not loci_data:
            return "[Spatial Loci Memory Index]\n(Empty - no memories retained yet)"

        lines = ["[Spatial Loci Memory Index]"]
        total_items = 0

        for wing, halls in sorted(loci_data.items()):
            for hall, keys in sorted(halls.items()):
                if not keys:
                    continue
                count = len(keys)
                total_items += count
                displayed_keys = keys[:max_items_per_hall]
                keys_str = ", ".join(displayed_keys)
                if count > max_items_per_hall:
                    keys_str += f", ... (+{count - max_items_per_hall} more)"

                unit = "key" if count == 1 else "keys"
                lines.append(f"- {wing}/{hall} ({count} {unit}): {keys_str}")

        if total_items == 0:
            return "[Spatial Loci Memory Index]\n(Empty - no memories retained yet)"

        return "\n".join(lines)
