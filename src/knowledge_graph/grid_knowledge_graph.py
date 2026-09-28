import networkx as nx


def build_grid_knowledge_graph():
    graph = nx.DiGraph()

    graph.add_node(
        "Harris County",
        node_type="location",
    )

    graph.add_node(
        "Hurricane Beryl",
        node_type="weather_event",
    )

    graph.add_node(
        "May 2024 Derecho",
        node_type="weather_event",
    )

    graph.add_node(
        "Major Outage",
        node_type="grid_event",
    )

    graph.add_node(
        "Human Engineering Review",
        node_type="governance_control",
    )

    graph.add_edge(
        "Harris County",
        "Hurricane Beryl",
        relation="experienced",
    )

    graph.add_edge(
        "Harris County",
        "May 2024 Derecho",
        relation="experienced",
    )

    graph.add_edge(
        "Hurricane Beryl",
        "Major Outage",
        relation="associated_with",
    )

    graph.add_edge(
        "May 2024 Derecho",
        "Major Outage",
        relation="associated_with",
    )

    graph.add_edge(
        "Major Outage",
        "Human Engineering Review",
        relation="requires_when_high_risk",
    )

    return graph


def get_graph_context(query: str) -> str:
    graph = build_grid_knowledge_graph()

    query_lower = query.lower()

    relevant_nodes = []

    for node in graph.nodes:

        if node.lower() in query_lower:
            relevant_nodes.append(node)

    if not relevant_nodes:

        if "hurricane" in query_lower:
            relevant_nodes.append(
                "Hurricane Beryl"
            )

        if "derecho" in query_lower:
            relevant_nodes.append(
                "May 2024 Derecho"
            )

        if "outage" in query_lower:
            relevant_nodes.append(
                "Major Outage"
            )

    statements = []

    for source in relevant_nodes:

        for _, target, data in graph.out_edges(
            source,
            data=True,
        ):

            relation = data[
                "relation"
            ]

            statements.append(
                f"{source} "
                f"{relation.replace('_', ' ')} "
                f"{target}."
            )

    if not statements:
        return (
            "No directly matching structured "
            "knowledge graph context was found."
        )

    return " ".join(statements)


if __name__ == "__main__":

    graph = build_grid_knowledge_graph()

    print(
        "Nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Edges:",
        graph.number_of_edges(),
    )

    print(
        get_graph_context(
            "What happened during Hurricane Beryl?"
        )
    )