#!/usr/bin/env python3
"""
Schema Evaluation Script for AutoG-S

Compares AutoG-S generated schema against ground truth metadata.
Evaluates precision/recall of relationships, extra tables, and statistics.

Usage:
    python evaluate_schema.py <ground_truth.yaml> <predicted.yaml> [output_path]
"""

import os
import yaml
import sys
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from contextlib import contextmanager
from Levenshtein import distance as lev_distance


@contextmanager
def tee_output(output_path: Optional[str] = None):
    """
    Context manager to duplicate print output to both console and file
    
    Args:
        output_path: Optional path to save output. If None, only prints to console.
    """
    if output_path is None:
        # No file output, just yield
        yield
        return
    
    # Open file for writing
    output_file = open(output_path, 'w', encoding='utf-8')
    
    # Save original print function
    original_print = print
    
    # Create new print function that writes to both console and file
    def dual_print(*args, **kwargs):
        # Print to console
        original_print(*args, **kwargs)
        # Print to file
        original_print(*args, **kwargs, file=output_file, flush=True)
    
    # Replace built-in print
    import builtins
    builtins.print = dual_print
    
    try:
        yield
    finally:
        # Restore original print
        builtins.print = original_print
        # Close file
        output_file.close()


class SchemaEvaluator:
    """Evaluates AutoG-S schema against ground truth"""
    
    def __init__(self, gt_path: str, pred_path: str):
        self.gt_path = gt_path
        self.pred_path = pred_path
        
        # Load metadata
        with open(gt_path, 'r') as f:
            self.gt_metadata = yaml.safe_load(f)
        
        with open(pred_path, 'r') as f:
            self.pred_metadata = yaml.safe_load(f)
        
        # Extract relationships
        self.gt_relationships = self._extract_relationships(self.gt_metadata)
        self.pred_relationships = self._extract_relationships(self.pred_metadata)
        
        # Resolve bridge tables in predictions
        bridge_resolved = self._resolve_bridge_tables(
            self.pred_metadata, 
            self.pred_relationships
        )
        
        # Identify bridge tables (tables with only PK and FKs)
        self.bridge_tables = self._identify_bridge_tables(self.pred_metadata)
        
        # Identify entity tables (tables with PK and few attributes)
        self.entity_tables = self._identify_entity_tables(self.pred_metadata)
        
        # Resolve entity tables in predictions (after bridge resolution)
        entity_resolved = self._resolve_entity_tables(
            self.pred_metadata,
            bridge_resolved
        )
        
        # Filter out relationships involving bridge/entity tables
        # Only keep relationships between original tables (not intermediate tables)
        self.resolved_pred_relationships = self._filter_intermediate_relationships(
            entity_resolved,
            self.bridge_tables,
            self.entity_tables
        )
        
        # Extract table information
        self.gt_tables = {t['name'] for t in self.gt_metadata['tables']}
        self.pred_tables = {t['name'] for t in self.pred_metadata['tables']}
    
    def _extract_relationships(self, metadata: Dict) -> Set[Tuple[str, str, str, str]]:
        """
        Extract all FK relationships from metadata
        
        Returns:
            Set of (source_table, source_col, target_table, target_col) tuples
        """
        relationships = set()
        
        for table in metadata['tables']:
            table_name = table['name']
            for col in table['columns']:
                if col['dtype'] == 'foreign_key' and 'link_to' in col:
                    link_to = col['link_to']
                    # Handle link_to format: "Table.Column"
                    parts = link_to.rsplit('.', 1)
                    if len(parts) == 2:
                        target_table, target_col = parts
                        relationships.add((table_name, col['name'], target_table, target_col))
        
        return relationships
    
    def _identify_bridge_tables(self, metadata: Dict) -> Set[str]:
        """
        Identify bridge/lookup tables (tables that start with "lkp_")
        
        Bridge tables are junction tables created by AutoG-S that:
        - Have names starting with "lkp_"
        - Typically have only PK and FK columns
        - Connect two other tables in a many-to-many relationship
        
        Returns:
            Set of bridge table names
        """
        bridge_tables = set()
        
        for table in metadata['tables']:
            table_name = table['name']
            
            # Bridge tables MUST start with "lkp_"
            if table_name.startswith('lkp_'):
                bridge_tables.add(table_name)
        
        return bridge_tables
    
    def _resolve_bridge_tables(
        self, 
        metadata: Dict, 
        relationships: Set[Tuple[str, str, str, str]]
    ) -> Set[Tuple[str, str, str, str]]:
        """
        Resolve relationships that go through bridge tables
        
        Bridge tables follow naming conventions:
        - Table name starts with "lkp_"
        - PK column starts with "lkp_" (e.g., "lkp_Accounts_Bank_IDID")
        - FK column ends with "_fk" (e.g., "Bank_ID_fk")
        
        For example:
            Accounts.Bank_ID -> lkp_Accounts_Bank_ID.lkp_Accounts_Bank_IDID (PK)
            lkp_Accounts_Bank_ID.Bank_ID_fk (FK) -> Bank.Bank_ID
        Should be resolved to:
            Accounts.Bank_ID -> Bank.Bank_ID
        
        Returns:
            Set of resolved relationships
        """
        resolved = set()
        bridge_tables = self._identify_bridge_tables(metadata)
        
        # Build incoming connections to bridge tables (pointing to PK)
        # Format: bridge_table -> [(src_table, src_col, bridge_pk_col)]
        bridge_incoming = defaultdict(list)
        for src_table, src_col, tgt_table, tgt_col in relationships:
            if tgt_table in bridge_tables:
                # Check if target column is the PK (starts with "lkp_")
                if tgt_col.startswith('lkp_'):
                    bridge_incoming[tgt_table].append((src_table, src_col, tgt_col))
        
        # Build outgoing connections from bridge tables (FK pointing out)
        # Format: bridge_table -> [(bridge_fk_col, tgt_table, tgt_col)]
        bridge_outgoing = defaultdict(list)
        for src_table, src_col, tgt_table, tgt_col in relationships:
            if src_table in bridge_tables:
                # Check if source column is the FK (ends with "_fk")
                if src_col.endswith('_fk'):
                    bridge_outgoing[src_table].append((src_col, tgt_table, tgt_col))
        
        # Resolve through bridge tables
        for bridge_table in bridge_tables:
            if bridge_table in bridge_incoming and bridge_table in bridge_outgoing:
                # Bridge table has both incoming and outgoing connections
                for src_table, src_col, bridge_pk in bridge_incoming[bridge_table]:
                    for bridge_fk, final_table, final_col in bridge_outgoing[bridge_table]:
                        # Create resolved relationship: src -> final
                        resolved.add((src_table, src_col, final_table, final_col))
        
        # Add non-bridge relationships
        for rel in relationships:
            src_table, src_col, tgt_table, tgt_col = rel
            if src_table not in bridge_tables and tgt_table not in bridge_tables:
                # Direct relationship, not through bridge
                resolved.add(rel)
        
        return resolved
    
    def _identify_entity_tables(self, metadata: Dict) -> Set[str]:
        """
        Identify entity tables created by AutoG-S normalization
        
        Entity tables must:
        1. NOT be in the original ground truth tables
        2. Have exactly 2 columns
        3. One column ends with "ID" (e.g., "AccountID", "Payment FormatID")
        4. Other column is the same name without "ID" (e.g., "Account", "Payment Format")
        
        Examples:
        - AccountID, Account
        - Payment FormatID, Payment Format
        - Receiving CurrencyID, Receiving Currency
        
        Returns:
            Set of entity table names
        """
        entity_tables = set()
        
        # Get ground truth table names for comparison
        gt_table_names = {t['name'].lower().strip() for t in self.gt_metadata['tables']}
        
        for table in metadata['tables']:
            table_name = table['name']
            columns = table['columns']
            
            # Must have exactly 2 columns
            if len(columns) != 2:
                continue
            
            # Must not be in ground truth (it's a generated entity table)
            if table_name.lower().strip() in gt_table_names:
                continue
            
            # Get column names
            col1_name = columns[0]['name']
            col2_name = columns[1]['name']
            
            # Check if one column ends with "ID" and the other is the same without "ID"
            # Try both orders
            if col1_name.endswith('ID'):
                # col1 is the ID column
                expected_name = col1_name[:-2].strip()  # Remove "ID"
                if col2_name.strip() == expected_name:
                    entity_tables.add(table_name)
            elif col2_name.endswith('ID'):
                # col2 is the ID column
                expected_name = col2_name[:-2].strip()  # Remove "ID"
                if col1_name.strip() == expected_name:
                    entity_tables.add(table_name)
        
        return entity_tables
    
    def _resolve_entity_tables(
        self,
        metadata: Dict,
        relationships: Set[Tuple[str, str, str, str]]
    ) -> Set[Tuple[str, str, str, str]]:
        """
        Resolve relationships that go through entity tables
        
        Pattern 1 (Chain): A.col1 -> C.pk, C.fk -> B.col2
            Resolves to: A.col1 -> B.col2
        
        Pattern 2 (Shared Entity): A.col1 -> C.pk, B.col2 -> C.pk
            Resolves to: A.col1 -> B.col2
            (Both tables reference the same entity, so they're related)
        
        Returns:
            Set of resolved relationships
        """
        resolved = set(relationships)  # Start with all relationships
        entity_tables = self._identify_entity_tables(metadata)
        
        # Build incoming connections to entity tables (tables that point TO entity)
        # Format: entity_table -> [(src_table, src_col, entity_col)]
        entity_incoming = defaultdict(list)
        for src_table, src_col, tgt_table, tgt_col in relationships:
            if tgt_table in entity_tables:
                entity_incoming[tgt_table].append((src_table, src_col, tgt_col))
        
        # Build outgoing connections from entity tables (entity points TO other tables)
        # Format: entity_table -> [(entity_col, tgt_table, tgt_col)]
        entity_outgoing = defaultdict(list)
        for src_table, src_col, tgt_table, tgt_col in relationships:
            if src_table in entity_tables:
                entity_outgoing[src_table].append((src_col, tgt_table, tgt_col))
        
        # Resolve through entity tables
        new_resolved = set()
        
        # Pattern 1: Chain through entity (A -> Entity -> B)
        for rel in resolved:
            src_table, src_col, tgt_table, tgt_col = rel
            
            # If this points to an entity table, try to resolve further
            if tgt_table in entity_tables and tgt_table in entity_outgoing:
                # Find where the entity table points to
                for entity_col, final_table, final_col in entity_outgoing[tgt_table]:
                    # Create resolved relationship: A.col1 -> B.col4
                    new_resolved.add((src_table, src_col, final_table, final_col))
            else:
                # Keep original relationship
                new_resolved.add(rel)
        
        # Pattern 2: Shared entity (A -> Entity <- B, resolve to A -> B)
        for entity_table in entity_tables:
            if entity_table in entity_incoming and len(entity_incoming[entity_table]) >= 2:
                # Multiple tables point to this entity
                incoming_list = entity_incoming[entity_table]
                
                # For each pair of tables pointing to the same entity
                for i, (table_a, col_a, entity_col_a) in enumerate(incoming_list):
                    for table_b, col_b, entity_col_b in incoming_list[i+1:]:
                        # If they point to the same column in the entity (usually the PK)
                        if entity_col_a == entity_col_b:
                            # Determine direction using semantic similarity to entity table
                            # The entity table name is usually similar to one of the original tables
                            # Direction: dissimilar_table -> similar_table
                            similarity_a = self._name_similarity(table_a, entity_table)
                            similarity_b = self._name_similarity(table_b, entity_table)
                            
                            if similarity_a > similarity_b:
                                # table_a more similar to entity -> B points to A
                                new_resolved.add((table_b, col_b, table_a, col_a))
                            elif similarity_b > similarity_a:
                                # table_b more similar to entity -> A points to B  
                                new_resolved.add((table_a, col_a, table_b, col_b))
                            else:
                                # Equal similarity, add both directions
                                new_resolved.add((table_a, col_a, table_b, col_b))
                                new_resolved.add((table_b, col_b, table_a, col_a))
        
        return new_resolved
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate lexical similarity between two table names using edit distance
        
        Uses normalized Levenshtein distance:
        - "Account" and "Accounts" -> high similarity
        - "Currency" and "Receiving Currency" -> high similarity
        - "Trans" and "Account" -> low similarity
        
        Returns:
            Similarity score between 0 and 1
        """
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        
        # Exact match
        if n1 == n2:
            return 1.0
        
        # Calculate Levenshtein similarity
        levenshtein_sim = 1 - (lev_distance(n1, n2) / max(len(n1), len(n2)))
        
        return levenshtein_sim
    
    def _filter_intermediate_relationships(
        self,
        relationships: Set[Tuple[str, str, str, str]],
        bridge_tables: Set[str],
        entity_tables: Set[str]
    ) -> Set[Tuple[str, str, str, str]]:
        """
        Filter out relationships that involve bridge or entity tables
        
        Only keep relationships between original tables (ground truth tables).
        This excludes intermediate relationships like:
        - A -> BridgeTable
        - EntityTable -> B
        - A -> EntityTable
        
        Args:
            relationships: Set of all relationships after resolution
            bridge_tables: Set of bridge table names
            entity_tables: Set of entity table names
            
        Returns:
            Filtered set of relationships excluding intermediate tables
        """
        intermediate_tables = bridge_tables | entity_tables
        
        filtered = set()
        for src_table, src_col, tgt_table, tgt_col in relationships:
            # Exclude if either source or target is an intermediate table
            if src_table not in intermediate_tables and tgt_table not in intermediate_tables:
                filtered.add((src_table, src_col, tgt_table, tgt_col))
        
        return filtered
    
    def _normalize_relationship(self, rel: Tuple[str, str, str, str]) -> Tuple[str, str, str, str]:
        """
        Normalize relationship for comparison
        Handles case sensitivity and whitespace
        """
        src_table, src_col, tgt_table, tgt_col = rel
        return (
            src_table.strip().lower(),
            src_col.strip().lower(),
            tgt_table.strip().lower(),
            tgt_col.strip().lower()
        )
    
    def evaluate_relationships(self) -> Dict:
        """
        Evaluate precision and recall of relationships
        
        Returns:
            Dict with precision, recall, F1, and detailed matches
        """
        # Normalize relationships for comparison
        gt_normalized = {self._normalize_relationship(r) for r in self.gt_relationships}
        pred_normalized = {self._normalize_relationship(r) for r in self.resolved_pred_relationships}
        
        # Find matches
        true_positives = gt_normalized & pred_normalized
        false_positives = pred_normalized - gt_normalized
        false_negatives = gt_normalized - pred_normalized
        
        # Calculate metrics
        precision = len(true_positives) / len(pred_normalized) if pred_normalized else 0
        recall = len(true_positives) / len(gt_normalized) if gt_normalized else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'gt_total': len(gt_normalized),
            'pred_total': len(pred_normalized),
            'tp_relationships': sorted(true_positives),
            'fp_relationships': sorted(false_positives),
            'fn_relationships': sorted(false_negatives)
        }
    
    def get_table_statistics(self) -> Dict:
        """
        Get statistics about tables and columns
        
        Returns:
            Dict with table counts, PK/FK counts per table, etc.
        """
        def count_column_types(metadata):
            stats = {}
            for table in metadata['tables']:
                table_name = table['name']
                pk_count = sum(1 for col in table['columns'] if col['dtype'] == 'primary_key')
                fk_count = sum(1 for col in table['columns'] if col['dtype'] == 'foreign_key')
                total_cols = len(table['columns'])
                
                stats[table_name] = {
                    'total_columns': total_cols,
                    'primary_keys': pk_count,
                    'foreign_keys': fk_count,
                    'other_columns': total_cols - pk_count - fk_count
                }
            return stats
        
        gt_stats = count_column_types(self.gt_metadata)
        pred_stats = count_column_types(self.pred_metadata)
        
        # Extra tables (not in ground truth)
        extra_tables = self.pred_tables - self.gt_tables
        missing_tables = self.gt_tables - self.pred_tables
        
        return {
            'gt_table_count': len(self.gt_tables),
            'pred_table_count': len(self.pred_tables),
            'extra_tables': sorted(extra_tables),
            'missing_tables': sorted(missing_tables),
            'bridge_tables': sorted(self.bridge_tables),
            'gt_table_stats': gt_stats,
            'pred_table_stats': pred_stats
        }
    
    def print_report(self):
        """Print comprehensive evaluation report"""
        print("=" * 80)
        print("SCHEMA EVALUATION REPORT")
        print("=" * 80)
        print(f"\nGround Truth: {self.gt_path}")
        print(f"Predicted:    {self.pred_path}")
        print()
        
        # Extracted relationships
        print("=" * 80)
        print("EXTRACTED RELATIONSHIPS")
        print("=" * 80)
        
        print(f"\nGround Truth Relationships ({len(self.gt_relationships)}):")
        for src_t, src_c, tgt_t, tgt_c in sorted(self.gt_relationships):
            print(f"  {src_t}.{src_c} → {tgt_t}.{tgt_c}")
        
        print(f"\nPredicted Relationships ({len(self.pred_relationships)}):")
        for src_t, src_c, tgt_t, tgt_c in sorted(self.pred_relationships):
            # Mark if it involves bridge or entity table
            markers = []
            if src_t in self.bridge_tables or tgt_t in self.bridge_tables:
                markers.append("BRIDGE")
            if src_t in self.entity_tables or tgt_t in self.entity_tables:
                markers.append("ENTITY")
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            print(f"  {src_t}.{src_c} → {tgt_t}.{tgt_c}{marker_str}")
        
        print(f"\nResolved Predicted Relationships ({len(self.resolved_pred_relationships)}):")
        print(f"  (After resolving {len(self.bridge_tables)} bridge tables and {len(self.entity_tables)} entity tables)")
        for src_t, src_c, tgt_t, tgt_c in sorted(self.resolved_pred_relationships):
            print(f"  {src_t}.{src_c} → {tgt_t}.{tgt_c}")
        
        # Relationship evaluation
        print("\n" + "=" * 80)
        print("RELATIONSHIP EVALUATION (Primary Keys & Foreign Keys)")
        print("=" * 80)
        
        rel_metrics = self.evaluate_relationships()
        
        print(f"\nGround Truth Relationships: {rel_metrics['gt_total']}")
        print(f"Predicted Relationships:    {rel_metrics['pred_total']}")
        print(f"  (After resolving {len(self.bridge_tables)} bridge tables and {len(self.entity_tables)} entity tables)")
        print()
        
        print(f"True Positives:  {rel_metrics['true_positives']}")
        print(f"False Positives: {rel_metrics['false_positives']}")
        print(f"False Negatives: {rel_metrics['false_negatives']}")
        print()
        
        print(f"Precision: {rel_metrics['precision']:.2%}")
        print(f"Recall:    {rel_metrics['recall']:.2%}")
        print(f"F1 Score:  {rel_metrics['f1']:.2%}")
        
        # Correctly identified relationships
        if rel_metrics['tp_relationships']:
            print(f"\n✓ Correctly Identified Relationships ({len(rel_metrics['tp_relationships'])}):")
            for src_t, src_c, tgt_t, tgt_c in rel_metrics['tp_relationships']:
                print(f"  {src_t}.{src_c} → {tgt_t}.{tgt_c}")
        
        # Missing relationships
        if rel_metrics['fn_relationships']:
            print(f"\n✗ Missing Relationships ({len(rel_metrics['fn_relationships'])}):")
            for src_t, src_c, tgt_t, tgt_c in rel_metrics['fn_relationships']:
                print(f"  {src_t}.{src_c} → {tgt_t}.{tgt_c}")
        
        # Extra relationships
        if rel_metrics['fp_relationships']:
            print(f"\n⚠ Extra Relationships Not in Ground Truth ({len(rel_metrics['fp_relationships'])}):")
            for src_t, src_c, tgt_t, tgt_c in rel_metrics['fp_relationships']:
                print(f"  {src_t}.{src_c} → {tgt_t}.{tgt_c}")
        
        # Table statistics
        print("\n" + "=" * 80)
        print("TABLE STATISTICS")
        print("=" * 80)
        
        table_stats = self.get_table_statistics()
        
        print(f"\nGround Truth Tables: {table_stats['gt_table_count']}")
        print(f"Predicted Tables:    {table_stats['pred_table_count']}")
        print(f"  Bridge Tables:     {len(table_stats['bridge_tables'])}")
        print(f"  Entity Tables:     {len(self.entity_tables)}")
        print()
        
        if table_stats['extra_tables']:
            print(f"Extra Tables ({len(table_stats['extra_tables'])}):")
            for table in table_stats['extra_tables']:
                table_type = ""
                if table in self.bridge_tables:
                    table_type = " [BRIDGE]"
                elif table in self.entity_tables:
                    table_type = " [ENTITY]"
                stats = table_stats['pred_table_stats'].get(table, {})
                print(f"  • {table}{table_type}")
                print(f"    Columns: {stats.get('total_columns', 0)} "
                      f"(PK: {stats.get('primary_keys', 0)}, "
                      f"FK: {stats.get('foreign_keys', 0)}, "
                      f"Other: {stats.get('other_columns', 0)})")
        
        if table_stats['missing_tables']:
            print(f"\nMissing Tables ({len(table_stats['missing_tables'])}):")
            for table in table_stats['missing_tables']:
                print(f"  • {table}")
        
        # Detailed table comparison
        print("\n" + "=" * 80)
        print("DETAILED TABLE COMPARISON")
        print("=" * 80)
        
        common_tables = self.gt_tables & self.pred_tables
        for table in sorted(common_tables):
            gt_stats = table_stats['gt_table_stats'][table]
            pred_stats = table_stats['pred_table_stats'][table]
            
            print(f"\nTable: {table}")
            print(f"  Ground Truth: {gt_stats['total_columns']} cols "
                  f"(PK: {gt_stats['primary_keys']}, FK: {gt_stats['foreign_keys']})")
            print(f"  Predicted:    {pred_stats['total_columns']} cols "
                  f"(PK: {pred_stats['primary_keys']}, FK: {pred_stats['foreign_keys']})")
            
            if gt_stats != pred_stats:
                print(f"  ⚠ Difference detected")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Relationship Precision: {rel_metrics['precision']:.2%}")
        print(f"Relationship Recall:    {rel_metrics['recall']:.2%}")
        print(f"Relationship F1:        {rel_metrics['f1']:.2%}")
        print(f"Extra Tables:           {len(table_stats['extra_tables'])}")
        print(f"  (Bridge Tables:       {len(table_stats['bridge_tables'])})")
        print("=" * 80)


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python evaluate_schema.py <ground_truth.yaml> <predicted.yaml> [output_dir]")
        print("\nArguments:")
        print("  ground_truth.yaml  - Path to ground truth metadata file")
        print("  predicted.yaml     - Path to predicted metadata file")
        print("  output_dir         - Optional directory to save evaluation_report.txt (default: print to console only)")
        print("\nExample:")
        print("  python evaluate_schema.py ./gt_metadata.yaml \\")
        print("                            ./final_metadata.yaml \\")
        print("                            ./output/")
        sys.exit(1)
    
    gt_path = sys.argv[1]
    pred_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) == 4 else None
    
    # Validate files exist
    if not Path(gt_path).exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        sys.exit(1)
    
    if not Path(pred_path).exists():
        print(f"Error: Predicted file not found: {pred_path}")
        sys.exit(1)
    
    # Prepare output file path
    output_file_path = None
    if output_dir:
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file_path = os.path.join(output_dir, "evaluation_report.txt")
    
    # Run evaluation with output redirection
    with tee_output(output_file_path):
        if output_file_path:
            print(f"Saving evaluation report to: {output_file_path}")
            print("=" * 80)
        
        evaluator = SchemaEvaluator(gt_path, pred_path)
        evaluator.print_report()
        
        if output_file_path:
            print("=" * 80)
            print(f"Report saved to: {output_file_path}")


if __name__ == "__main__":
    main()
