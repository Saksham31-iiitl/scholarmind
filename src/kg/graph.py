"""
Neo4j knowledge-graph builder.

Schema:
  (:Paper {id, title, year, url})
  (:Author {name})
  (:Concept {name})
  (:Claim {text, paper_id})

Relations:
  (Author)-[:WROTE]->(Paper)
  (Paper)-[:MENTIONS]->(Concept)
  (Paper)-[:STATES]->(Claim)
  (Paper)-[:CITES]->(Paper)
"""
from __future__ import annotations
from contextlib import contextmanager
from neo4j import GraphDatabase, Driver
from src.utils.config import settings
from src.utils.logging import logger


class KnowledgeGraph:
    def __init__(self) -> None:
        user = settings.neo4j_username or settings.neo4j_user
        self.driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    @contextmanager
    def _session(self):
        with self.driver.session() as s:
            yield s

    # ----- schema -----
    def init_schema(self) -> None:
        stmts = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
        ]
        with self._session() as s:
            for q in stmts:
                s.run(q)
        logger.info("Neo4j schema initialized")

    # ----- ingest -----
    def add_paper(self, paper: dict, concepts: list[str], claims: list[str]) -> None:
        with self._session() as s:
            s.run(
                """
                MERGE (p:Paper {id:$id})
                SET p.title=$title, p.url=$url, p.published=$published
                """,
                id=paper["id"], title=paper["title"],
                url=paper.get("url"), published=paper.get("published"),
            )
            for author in paper.get("authors", []):
                s.run(
                    """
                    MERGE (a:Author {name:$name})
                    MERGE (p:Paper {id:$id})
                    MERGE (a)-[:WROTE]->(p)
                    """,
                    name=author, id=paper["id"],
                )
            for concept in concepts:
                s.run(
                    """
                    MERGE (c:Concept {name:$name})
                    MERGE (p:Paper {id:$id})
                    MERGE (p)-[:MENTIONS]->(c)
                    """,
                    name=concept.lower().strip(), id=paper["id"],
                )
            for claim in claims:
                s.run(
                    """
                    MATCH (p:Paper {id:$id})
                    CREATE (cl:Claim {text:$text, paper_id:$id})
                    MERGE (p)-[:STATES]->(cl)
                    """,
                    id=paper["id"], text=claim,
                )

    # ----- query -----
    def find_related_papers(self, concept: str, limit: int = 10) -> list[dict]:
        cypher = """
        MATCH (p:Paper)-[:MENTIONS]->(c:Concept {name:$name})
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
        RETURN p.id AS id, p.title AS title, p.url AS url,
               collect(DISTINCT a.name) AS authors
        LIMIT $limit
        """
        with self._session() as s:
            return [dict(r) for r in s.run(cypher, name=concept.lower(), limit=limit)]

    def graph_summary(self) -> dict:
        with self._session() as s:
            counts = s.run("""
                MATCH (p:Paper) WITH count(p) AS papers
                MATCH (a:Author) WITH papers, count(a) AS authors
                MATCH (c:Concept) WITH papers, authors, count(c) AS concepts
                MATCH (cl:Claim) RETURN papers, authors, concepts, count(cl) AS claims
            """).single()
            return dict(counts) if counts else {}
