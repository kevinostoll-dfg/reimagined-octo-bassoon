#!/usr/bin/env python3
"""
TSLA Data Extraction Test Script

This script queries the Memgraph database to extract all data related to TSLA (Tesla)
based on the graph schema. It explores:
- TSLA as an organization
- Statements made by TSLA
- People associated with TSLA
- Concepts, dates, metrics mentioned with TSLA
- Risks, relationships, and connections
"""

import sys
import os
from neo4j import GraphDatabase
from tabulate import tabulate
import json
from datetime import datetime

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config


def connect_to_memgraph():
    """Connect to Memgraph database."""
    driver = GraphDatabase.driver(
        config.MEMGRAPH_URI,
        auth=(config.MEMGRAPH_USER, config.MEMGRAPH_PASSWORD)
    )
    return driver


def find_tsla_org(session):
    """Find TSLA organization node(s)."""
    print("\n" + "="*80)
    print("1. FINDING TSLA ORGANIZATION NODE(S)")
    print("="*80)
    
    # Try multiple ways to find TSLA
    queries = [
        {
            "name": "By canonical_name = 'TSLA'",
            "query": "MATCH (org:ORG {canonical_name: 'TSLA'}) RETURN org"
        },
        {
            "name": "By canonical_name containing 'Tesla'",
            "query": "MATCH (org:ORG) WHERE org.canonical_name CONTAINS 'Tesla' OR org.canonical_name CONTAINS 'TSLA' RETURN org"
        },
        {
            "name": "By variants containing 'TSLA'",
            "query": "MATCH (org:ORG) WHERE 'TSLA' IN org.variants OR org.variants CONTAINS 'TSLA' RETURN org"
        },
        {
            "name": "Case-insensitive search",
            "query": "MATCH (org:ORG) WHERE toLower(org.canonical_name) CONTAINS 'tesla' OR toLower(org.canonical_name) CONTAINS 'tsla' RETURN org LIMIT 10"
        }
    ]
    
    tsla_nodes = []
    for q in queries:
        result = session.run(q["query"]).data()
        if result:
            print(f"\n✅ Found via: {q['name']}")
            for record in result:
                org = record['org']
                tsla_nodes.append(org)
                print(f"  - canonical_name: {org.get('canonical_name', 'N/A')}")
                print(f"  - entity_id: {org.get('entity_id', 'N/A')}")
                print(f"  - variants: {org.get('variants', 'N/A')}")
                print(f"  - mention_count: {org.get('mention_count', 'N/A')}")
            break
    
    if not tsla_nodes:
        print("\n⚠️  No TSLA organization found. Trying broader search...")
        # Get all ORG nodes to see what's available
        all_orgs = session.run("MATCH (org:ORG) RETURN org.canonical_name as name LIMIT 20").data()
        if all_orgs:
            print("\nSample organizations in database:")
            for org in all_orgs[:10]:
                print(f"  - {org['name']}")
    
    return tsla_nodes


def get_tsla_statements(session, tsla_canonical_name):
    """Get all statements that mention TSLA."""
    print("\n" + "="*80)
    print("2. STATEMENTS MENTIONING TSLA")
    print("="*80)
    
    # Updated queries based on working patterns
    queries = [
        {
            "name": "Statements mentioning Tesla in text (most reliable)",
            "query": """
                MATCH (s:STATEMENT)
                WHERE s.text CONTAINS 'Tesla' OR s.text CONTAINS 'TSLA' OR s.text CONTAINS 'tesla'
                OPTIONAL MATCH (person:PERSON)-[:SAID]->(s)
                WITH DISTINCT s, COLLECT(DISTINCT person.canonical_name) as speakers
                RETURN s.statement_id as statement_id,
                       s.text as text,
                       speakers[0] as primary_speaker,
                       speakers as all_speakers,
                       SIZE(speakers) as speaker_count
                ORDER BY s.statement_id
                LIMIT 50
            """
        },
        {
            "name": "Statements from Tesla company sections",
            "query": """
                MATCH (section:SECTION)
                WHERE section.ticker = 'TSLA' OR section.company_name CONTAINS 'Tesla'
                MATCH (s:STATEMENT)
                WHERE s.statement_id STARTS WITH section.section_id
                OPTIONAL MATCH (person:PERSON)-[:SAID]->(s)
                WITH DISTINCT s, section, COLLECT(DISTINCT person.canonical_name) as speakers
                RETURN s.statement_id as statement_id,
                       s.text as text,
                       speakers[0] as primary_speaker,
                       speakers as all_speakers,
                       section.company_name as company
                ORDER BY s.statement_id
                LIMIT 50
            """
        },
        {
            "name": "Statements via SVO_TRIPLE mentioning Tesla",
            "query": """
                MATCH (tesla:ORG {canonical_name: $name})<-[r:SVO_TRIPLE]-(s:STATEMENT)
                OPTIONAL MATCH (person:PERSON)-[:SAID]->(s)
                WITH DISTINCT s, r, COLLECT(DISTINCT person.canonical_name) as speakers
                RETURN s.statement_id as statement_id,
                       s.text as text,
                       speakers[0] as primary_speaker,
                       speakers as all_speakers,
                       r.verb_text as verb
                ORDER BY s.statement_id
                LIMIT 50
            """
        },
        {
            "name": "Statements sharing Tesla-related entities (filtered)",
            "query": """
                MATCH (tesla:ORG {canonical_name: $name})-[r1:CO_MENTIONED]-(shared)
                MATCH (s:STATEMENT)-[r2:CO_MENTIONED|SAID]-(shared)
                WHERE shared <> tesla
                  AND (shared.canonical_name CONTAINS 'Tesla' 
                       OR s.text CONTAINS 'Tesla' 
                       OR s.text CONTAINS 'TSLA')
                OPTIONAL MATCH (person:PERSON)-[:SAID]->(s)
                WITH DISTINCT s, shared, COLLECT(DISTINCT person.canonical_name) as speakers
                RETURN s.statement_id as statement_id,
                       s.text as text,
                       speakers[0] as primary_speaker,
                       speakers as all_speakers,
                       labels(shared)[0] as shared_type,
                       shared.canonical_name as shared_name
                ORDER BY s.statement_id
                LIMIT 50
            """
        }
    ]
    
    all_statements = []
    seen_ids = set()  # Track seen statement IDs to avoid duplicates
    
    for q in queries:
        try:
            result = session.run(q["query"], name=tsla_canonical_name).data()
            if result:
                print(f"\n✅ Found via: {q['name']}")
                print(f"Found {len(result)} unique statements:")
                statements_table = []
                
                for record in result:
                    stmt_id = record['statement_id']
                    # Skip if we've already seen this statement
                    if stmt_id in seen_ids:
                        continue
                    seen_ids.add(stmt_id)
                    
                    text = record['text']
                    if len(text) > 100:
                        text = text[:97] + "..."
                    
                    speaker = record.get('primary_speaker') or 'Unknown'
                    all_speakers = record.get('all_speakers', [])
                    speaker_count = record.get('speaker_count', len(all_speakers))
                    
                    # Show speaker info
                    if speaker_count > 1:
                        speaker_display = f"{speaker} (+{speaker_count-1} others)"
                    else:
                        speaker_display = speaker
                    
                    statements_table.append([
                        len(statements_table) + 1,
                        stmt_id,
                        speaker_display,
                        text
                    ])
                    all_statements.append(record)
                
                if statements_table:
                    print(tabulate(statements_table[:20], headers=["#", "Statement ID", "Speaker", "Text"], tablefmt="simple"))
                    if len(statements_table) > 20:
                        print(f"\n... and {len(statements_table) - 20} more statements")
                
                # Stop after first successful query that returns results
                if all_statements:
                    break
        except Exception as e:
            print(f"\n⚠️  Query '{q['name']}' failed: {e}")
            continue
    
    if not all_statements:
        print("\n⚠️  No statements found mentioning Tesla")
    
    return all_statements


def get_tsla_people(session, tsla_canonical_name):
    """Get people associated with TSLA."""
    print("\n" + "="*80)
    print("3. PEOPLE ASSOCIATED WITH TSLA")
    print("="*80)
    
    queries = [
        {
            "name": "People who made statements mentioning Tesla (text-based)",
            "query": """
                MATCH (s:STATEMENT)
                WHERE s.text CONTAINS 'Tesla' OR s.text CONTAINS 'TSLA' OR s.text CONTAINS 'tesla'
                MATCH (p:PERSON)-[:SAID]->(s)
                RETURN p.canonical_name as person, 
                       COUNT(DISTINCT s.statement_id) as statement_count
                ORDER BY statement_count DESC
                LIMIT 30
            """
        },
        {
            "name": "People with roles who mentioned Tesla",
            "query": """
                MATCH (s:STATEMENT)
                WHERE s.text CONTAINS 'Tesla' OR s.text CONTAINS 'TSLA'
                MATCH (p:PERSON)-[:SAID]->(s)
                OPTIONAL MATCH (p)-[:HAS_ROLE]->(r:ROLE)
                WITH p, r, COUNT(DISTINCT s.statement_id) as statement_count
                RETURN p.canonical_name as person, 
                       COLLECT(DISTINCT r.title)[0] as role,
                       statement_count
                ORDER BY statement_count DESC
                LIMIT 30
            """
        },
        {
            "name": "People connected via Tesla-related entities",
            "query": """
                MATCH (tesla:ORG {canonical_name: $name})-[r1:CO_MENTIONED]-(shared)
                MATCH (s:STATEMENT)-[r2:CO_MENTIONED|SAID]-(shared)
                WHERE shared <> tesla
                  AND (shared.canonical_name CONTAINS 'Tesla' 
                       OR s.text CONTAINS 'Tesla' 
                       OR s.text CONTAINS 'TSLA')
                MATCH (p:PERSON)-[:SAID]->(s)
                RETURN p.canonical_name as person, 
                       COUNT(DISTINCT s.statement_id) as statement_count
                ORDER BY statement_count DESC
                LIMIT 30
            """
        }
    ]
    
    all_people = []
    seen_people = set()
    
    for q in queries:
        try:
            result = session.run(q["query"], name=tsla_canonical_name).data()
            if result:
                print(f"\n{q['name']}:")
                people_table = []
                for record in result:
                    person = record.get('person', 'N/A')
                    if person in seen_people:
                        continue
                    seen_people.add(person)
                    
                    if 'role' in record and record.get('role'):
                        role = record.get('role', 'N/A')
                        count = record.get('statement_count', 0)
                        people_table.append([person, f"{role} ({count} statements)"])
                    else:
                        count = record.get('statement_count', 0)
                        people_table.append([person, f"{count} statements"])
                    all_people.append(person)
                
                if people_table:
                    print(tabulate(people_table, headers=["Person", "Details"], tablefmt="simple"))
                
                # Stop after first successful query
                if all_people:
                    break
        except Exception as e:
            print(f"\n⚠️  Query '{q['name']}' failed: {e}")
            continue
    
    if not all_people:
        print("\n⚠️  No people found")
    
    return all_people


def get_tsla_mentioned_entities(session, tsla_canonical_name):
    """Get entities mentioned with TSLA."""
    print("\n" + "="*80)
    print("4. ENTITIES MENTIONED WITH TSLA")
    print("="*80)
    
    # Get co-mentioned entities
    query = """
        MATCH (org:ORG {canonical_name: $name})-[:CO_MENTIONED]->(entity)
        RETURN labels(entity)[0] as entity_type,
               entity.canonical_name as name,
               COUNT(entity) as mention_count
        ORDER BY mention_count DESC
        LIMIT 30
    """
    
    result = session.run(query, name=tsla_canonical_name).data()
    
    if result:
        print("\nCo-mentioned entities:")
        entities_table = []
        for record in result:
            entities_table.append([
                record['entity_type'],
                record['name'],
                record['mention_count']
            ])
        print(tabulate(entities_table, headers=["Type", "Name", "Mention Count"], tablefmt="simple"))
    else:
        print("\n⚠️  No co-mentioned entities found")
    
    return result


def get_tsla_dates(session, tsla_canonical_name):
    """Get dates/temporal context for TSLA."""
    print("\n" + "="*80)
    print("5. DATES & TEMPORAL CONTEXT FOR TSLA")
    print("="*80)
    
    query = """
        MATCH (org:ORG {canonical_name: $name})-[:TEMPORAL_CONTEXT|CO_MENTIONED]->(date:DATE)
        RETURN date.canonical_name as date, COUNT(date) as count
        ORDER BY date DESC
        LIMIT 20
    """
    
    result = session.run(query, name=tsla_canonical_name).data()
    
    if result:
        print("\nDates associated with TSLA:")
        dates_table = []
        for record in result:
            dates_table.append([record['date'], record['count']])
        print(tabulate(dates_table, headers=["Date", "Count"], tablefmt="simple"))
    else:
        print("\n⚠️  No dates found")
    
    return result


def get_tsla_metrics(session, tsla_canonical_name):
    """Get metrics related to TSLA."""
    print("\n" + "="*80)
    print("6. METRICS & QUANTITIES FOR TSLA")
    print("="*80)
    
    query = """
        MATCH (org:ORG {canonical_name: $name})-[:QUANTITY_OF|SVO_TRIPLE]->(metric)
        WHERE metric:METRIC OR metric:MONEY OR metric:PERCENT OR metric:CARDINAL
        RETURN labels(metric)[0] as metric_type,
               metric.canonical_name as value,
               metric.value as metric_value,
               COUNT(metric) as count
        ORDER BY count DESC
        LIMIT 30
    """
    
    result = session.run(query, name=tsla_canonical_name).data()
    
    if result:
        print("\nMetrics associated with TSLA:")
        metrics_table = []
        for record in result:
            value = record.get('metric_value') or record.get('value', 'N/A')
            metrics_table.append([
                record['metric_type'],
                record['value'],
                value,
                record['count']
            ])
        print(tabulate(metrics_table, headers=["Type", "Canonical Name", "Value", "Count"], tablefmt="simple"))
    else:
        print("\n⚠️  No metrics found")
    
    return result


def get_tsla_risks(session, tsla_canonical_name):
    """Get risks associated with TSLA."""
    print("\n" + "="*80)
    print("7. RISKS ASSOCIATED WITH TSLA")
    print("="*80)
    
    queries = [
        {
            "name": "Risks TSLA is exposed to",
            "query": """
                MATCH (org:ORG {canonical_name: $name})-[:EXPOSES_TO_RISK]->(risk:RISK)
                RETURN risk.canonical_name as risk_name, COUNT(risk) as count
                ORDER BY count DESC
                LIMIT 20
            """
        },
        {
            "name": "Risks TSLA mitigates",
            "query": """
                MATCH (org:ORG {canonical_name: $name})-[:MITIGATES_RISK]->(risk:RISK)
                RETURN risk.canonical_name as risk_name, COUNT(risk) as count
                ORDER BY count DESC
                LIMIT 20
            """
        }
    ]
    
    all_risks = []
    for q in queries:
        result = session.run(q["query"], name=tsla_canonical_name).data()
        if result:
            print(f"\n{q['name']}:")
            risks_table = []
            for record in result:
                risks_table.append([record['risk_name'], record['count']])
                all_risks.append(record['risk_name'])
            print(tabulate(risks_table, headers=["Risk", "Count"], tablefmt="simple"))
    
    if not all_risks:
        print("\n⚠️  No risks found")
    
    return all_risks


def get_tsla_concepts(session, tsla_canonical_name):
    """Get concepts related to TSLA."""
    print("\n" + "="*80)
    print("8. CONCEPTS RELATED TO TSLA")
    print("="*80)
    
    query = """
        MATCH (org:ORG {canonical_name: $name})-[:CO_MENTIONED|SVO_TRIPLE]->(concept:CONCEPT)
        RETURN concept.canonical_name as concept_name, COUNT(concept) as count
        ORDER BY count DESC
        LIMIT 30
    """
    
    result = session.run(query, name=tsla_canonical_name).data()
    
    if result:
        print("\nConcepts associated with TSLA:")
        concepts_table = []
        for record in result:
            concepts_table.append([record['concept_name'], record['count']])
        print(tabulate(concepts_table, headers=["Concept", "Count"], tablefmt="simple"))
    else:
        print("\n⚠️  No concepts found")
    
    return result


def get_tsla_sections(session, tsla_canonical_name):
    """Get sections/filings related to TSLA."""
    print("\n" + "="*80)
    print("9. SECTIONS/FILINGS FOR TSLA")
    print("="*80)
    
    query = """
        MATCH (section:SECTION)
        WHERE section.ticker = 'TSLA' OR section.company_name CONTAINS 'Tesla'
        RETURN section.section_name as section_name,
               section.ticker as ticker,
               section.company_name as company_name,
               section.filing_date as filing_date,
               section.period_of_report as period
        ORDER BY section.filing_date DESC
        LIMIT 20
    """
    
    result = session.run(query).data()
    
    if result:
        print("\nSections/Filings:")
        sections_table = []
        for record in result:
            sections_table.append([
                record['section_name'],
                record['ticker'],
                record['company_name'],
                record['filing_date'],
                record['period']
            ])
        print(tabulate(sections_table, headers=["Section", "Ticker", "Company", "Filing Date", "Period"], tablefmt="simple"))
    else:
        print("\n⚠️  No sections found")
    
    return result


def get_tsla_relationships_summary(session, tsla_canonical_name):
    """Get summary of all relationship types for TSLA."""
    print("\n" + "="*80)
    print("10. RELATIONSHIP SUMMARY FOR TSLA")
    print("="*80)
    
    query = """
        MATCH (org:ORG {canonical_name: $name})-[r]->(connected)
        RETURN type(r) as relationship_type,
               labels(connected)[0] as target_type,
               COUNT(r) as count
        ORDER BY count DESC
    """
    
    result = session.run(query, name=tsla_canonical_name).data()
    
    if result:
        print("\nRelationship summary:")
        rels_table = []
        for record in result:
            rels_table.append([
                record['relationship_type'],
                record['target_type'],
                record['count']
            ])
        print(tabulate(rels_table, headers=["Relationship", "Target Type", "Count"], tablefmt="simple"))
    else:
        print("\n⚠️  No relationships found")
    
    return result


def main():
    """Main function to run TSLA data extraction."""
    print("\n" + "="*80)
    print("TSLA DATA EXTRACTION TEST")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    driver = connect_to_memgraph()
    results = {}
    
    try:
        with driver.session() as session:
            # 1. Find TSLA organization
            tsla_nodes = find_tsla_org(session)
            results['tsla_nodes'] = [dict(node) for node in tsla_nodes]
            
            if not tsla_nodes:
                print("\n❌ TSLA organization not found. Cannot proceed with data extraction.")
                print("Please check:")
                print("  1. TSLA data exists in the database")
                print("  2. The canonical_name or variants match 'TSLA' or 'Tesla'")
                return
            
            # Use first TSLA node found
            tsla_canonical_name = tsla_nodes[0].get('canonical_name')
            print(f"\n✅ Using TSLA node: {tsla_canonical_name}")
            
            # 2. Get statements
            statements = get_tsla_statements(session, tsla_canonical_name)
            results['statements'] = statements
            
            # 3. Get people
            people = get_tsla_people(session, tsla_canonical_name)
            results['people'] = people
            
            # 4. Get mentioned entities
            entities = get_tsla_mentioned_entities(session, tsla_canonical_name)
            results['mentioned_entities'] = entities
            
            # 5. Get dates
            dates = get_tsla_dates(session, tsla_canonical_name)
            results['dates'] = dates
            
            # 6. Get metrics
            metrics = get_tsla_metrics(session, tsla_canonical_name)
            results['metrics'] = metrics
            
            # 7. Get risks
            risks = get_tsla_risks(session, tsla_canonical_name)
            results['risks'] = risks
            
            # 8. Get concepts
            concepts = get_tsla_concepts(session, tsla_canonical_name)
            results['concepts'] = concepts
            
            # 9. Get sections
            sections = get_tsla_sections(session, tsla_canonical_name)
            results['sections'] = sections
            
            # 10. Get relationship summary
            relationships = get_tsla_relationships_summary(session, tsla_canonical_name)
            results['relationships'] = relationships
            
            print("\n" + "="*80)
            print("TSLA DATA EXTRACTION COMPLETE")
            print("="*80)
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Summary
            print("="*80)
            print("SUMMARY")
            print("="*80)
            print(f"✅ Found {len(tsla_nodes)} Tesla organization node(s)")
            print(f"{'✅' if statements else '⚠️ '} Found {len(statements) if statements else 0} statements")
            print(f"{'✅' if people else '⚠️ '} Found {len(people) if people else 0} people")
            print(f"{'✅' if entities else '⚠️ '} Found {len(entities) if entities else 0} co-mentioned entities")
            print(f"{'✅' if dates else '⚠️ '} Found {len(dates) if dates else 0} dates")
            print(f"{'✅' if metrics else '⚠️ '} Found {len(metrics) if metrics else 0} metrics")
            print(f"{'✅' if risks else '⚠️ '} Found {len(risks) if risks else 0} risks")
            print(f"{'✅' if concepts else '⚠️ '} Found {len(concepts) if concepts else 0} concepts")
            print(f"{'✅' if sections else '⚠️ '} Found {len(sections) if sections else 0} sections/filings")
            print(f"{'✅' if relationships else '⚠️ '} Found {len(relationships) if relationships else 0} relationship types")
            print("="*80)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()


if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate for pretty printing...")
        import subprocess
        subprocess.check_call(["python", "-m", "pip", "install", "tabulate"])
    
    main()

