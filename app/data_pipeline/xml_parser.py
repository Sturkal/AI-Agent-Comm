"""Utilities for parsing SAP SuccessFactors Incentive Management XML exports.

The parser focuses on the parts that are most useful for downstream RAG:
rule identity, human-readable description, referenced variables/inputs, and the
core formula or condition logic flattened into text that an LLM can read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET


def _local_name(tag: str) -> str:
    """Strip namespaces from XML tag names."""

    if "}" in tag:
        return tag.split("}", 1)[1].lower()
    return tag.lower()


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _node_text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    parts = [part.strip() for part in node.itertext() if part and part.strip()]
    return _normalize_text(" ".join(parts))


def _first_non_empty(values: Iterable[str]) -> str:
    for value in values:
        normalized = _normalize_text(value)
        if normalized:
            return normalized
    return ""


def _extract_attributes(node: ET.Element) -> Dict[str, str]:
    return {key.lower(): _normalize_text(value) for key, value in node.attrib.items()}


def _get_attribute(node: ET.Element, *names: str) -> str:
    lower_map = {key.lower(): value for key, value in node.attrib.items()}
    for name in names:
        value = _normalize_text(lower_map.get(name.lower()))
        if value:
            return value
    return ""


@dataclass
class ParsedRule:
    record_type: str
    rule_name: str
    description: str
    variables: List[str]
    references: List[str]
    formula: str
    pseudo_code: str
    metadata: Dict[str, Any]


class SAPXMLParser:
    """Parse SAP SFIM XML rules into flat, LLM-friendly objects."""

    RECORD_TAGS = {"rule", "formula", "plan"}
    EXPRESSION_TAGS = {"action_expression", "formula_expression", "condition_expression", "rollup_expression"}
    VARIABLE_TAGS = {
        "variable",
        "input",
        "parameter",
        "operand",
        "reference",
        "field",
        "measure",
        "attribute",
        "data_field",
    }
    DESCRIPTION_TAGS = {"description", "desc", "summary", "comment"}
    NAME_ATTRS = ("name", "id", "key", "label", "rule_name")

    def parse_file(self, file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)
        return self.parse_string(path.read_text(encoding="utf-8"))

    def parse_string(self, xml_content: str) -> List[Dict[str, Any]]:
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            # SAP exports occasionally contain formatting artifacts or wrapped
            # fragments. BeautifulSoup can salvage many of those cases.
            return self._parse_with_bs4(xml_content)

        rules = self._collect_rules(root)
        return [asdict(rule) for rule in rules]

    def parse_folder(self, folder_path: str | Path) -> List[Dict[str, Any]]:
        folder = Path(folder_path)
        results: List[Dict[str, Any]] = []
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() == ".xml":
                results.extend(self.parse_file(path))
        return results

    def _parse_with_bs4(self, xml_content: str) -> List[Dict[str, Any]]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(xml_content, "xml")
        results: List[Dict[str, Any]] = []

        for node in soup.find_all(True):
            if _local_name(getattr(node, "name", "")) not in self.RECORD_TAGS:
                continue
            rule_name = self._extract_name_bs4(node)
            description = self._extract_description_bs4(node)
            variables = self._extract_variables_bs4(node)
            references = self._extract_references_bs4(node)
            formula = self._extract_formula_bs4(node)
            pseudo_code = self._build_pseudo_code(getattr(node, "name", ""), rule_name, variables, references, formula)
            results.append(
                asdict(
                    ParsedRule(
                        record_type=_local_name(getattr(node, "name", "")),
                        rule_name=rule_name,
                        description=description,
                        variables=variables,
                        references=references,
                        formula=formula,
                        pseudo_code=pseudo_code,
                        metadata={"source": "bs4_fallback", "tag": getattr(node, "name", "")},
                    )
                )
            )

        return results

    def _collect_rules(self, root: ET.Element) -> List[ParsedRule]:
        candidates = self._iter_rule_nodes(root)
        rules: List[ParsedRule] = []

        for node in candidates:
            record_type = _local_name(node.tag)
            rule_name = self._extract_rule_name(node)
            description = self._extract_description(node)
            variables = self._extract_variables(node)
            references = self._extract_references(node)
            formula = self._extract_formula(node)
            pseudo_code = self._build_pseudo_code(record_type, rule_name, variables, references, formula)
            rules.append(
                ParsedRule(
                    record_type=record_type,
                    rule_name=rule_name,
                    description=description,
                    variables=variables,
                    references=references,
                    formula=formula,
                    pseudo_code=pseudo_code,
                    metadata=self._extract_metadata(node),
                )
            )

        return rules

    def _iter_rule_nodes(self, root: ET.Element) -> List[ET.Element]:
        nodes = []
        for node in root.iter():
            if _local_name(node.tag) in self.RECORD_TAGS:
                nodes.append(node)

        # If the document has no obvious rule wrapper, treat the root as a rule
        # candidate so the parser still emits something useful.
        if not nodes:
            nodes.append(root)
        return nodes

    def _extract_rule_name(self, node: ET.Element) -> str:
        attr_name = _get_attribute(node, *self.NAME_ATTRS)
        if attr_name:
            return attr_name

        for child in node.iter():
            tag = _local_name(child.tag)
            if tag in {"name", "rulename", "rule_name", "title"}:
                text = _node_text(child)
                if text:
                    return text

        return ""

    def _extract_description(self, node: ET.Element) -> str:
        attr_description = _get_attribute(node, "description", "desc", "summary", "comment")
        if attr_description:
            return attr_description

        for child in node.iter():
            if _local_name(child.tag) in self.DESCRIPTION_TAGS:
                text = _node_text(child)
                if text:
                    return text
        return ""

    def _extract_references(self, node: ET.Element) -> List[str]:
        references: List[str] = []
        seen: set[str] = set()
        for child in node.iter():
            tag = _local_name(child.tag)
            if tag in {
                "rule_element_ref",
                "component_ref",
                "formula_ref",
                "plan_ref",
                "measurement_ref",
                "incentive_ref",
                "output_reference",
                "mdltvar_ref",
            }:
                candidate = _get_attribute(child, "name", "id", "key", "label", "rule", "rule_name", "ref")
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    references.append(candidate)
        return references

    def _extract_variables(self, node: ET.Element) -> List[str]:
        variables: List[str] = []
        seen: set[str] = set()

        for child in node.iter():
            tag = _local_name(child.tag)
            if tag in self.VARIABLE_TAGS:
                candidate = self._extract_variable_name(child)
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    variables.append(candidate)
                    continue

            # Fallback: capture common XML attribute names that often carry
            # variable references in SAP rule exports.
            for attr_name in ("variable", "input", "ref", "refid", "field", "source"):
                candidate = _get_attribute(child, attr_name)
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    variables.append(candidate)

        return variables

    def _extract_variable_name(self, node: ET.Element) -> str:
        preferred_attrs = ("name", "id", "ref", "refid", "key", "code", "path")
        for attr in preferred_attrs:
            value = _get_attribute(node, attr)
            if value:
                return value

        text = _node_text(node)
        if text and len(text) <= 120:
            return text
        return ""

    def _extract_formula(self, node: ET.Element) -> str:
        formula_fragments = self._collect_formula_fragments(node)
        if formula_fragments:
            return self._dedupe_join(formula_fragments)

        # Fallback to the full node text if we did not find a structured formula.
        return _node_text(node)

    def _collect_formula_fragments(self, node: ET.Element) -> List[str]:
        tag = _local_name(node.tag)

        if tag in self.EXPRESSION_TAGS:
            fragments: List[str] = []
            for child in list(node):
                fragments.extend(self._collect_formula_fragments(child))
            if fragments:
                return fragments
            text = _node_text(node)
            return [text] if text else []

        if tag in self._formula_node_tags():
            fragment = self._format_formula_node(node)
            return [fragment] if fragment else []

        fragments = []
        for child in list(node):
            fragments.extend(self._collect_formula_fragments(child))
        if fragments:
            return fragments
        text = _node_text(node)
        return [text] if text else []

    def _format_formula_node(self, node: ET.Element) -> str:
        tag = _local_name(node.tag)
        label = tag.upper()
        attrs = _extract_attributes(node)
        text = _node_text(node)
        name = _get_attribute(node, "name", "id", "key", "label", "ref", "refid", "code", "type")
        child_texts = [
            fragment
            for child in list(node)
            for fragment in self._collect_formula_fragments(child)
            if fragment
        ]

        if tag == "function":
            extra_attrs = self._compact_formula_attrs(attrs, exclude={"name", "id", "key", "label", "ref", "refid", "code", "type"})
            if name and child_texts:
                return f"FUNCTION {name}{extra_attrs}({', '.join(child_texts)})"
            if name:
                return f"FUNCTION {name}{extra_attrs}"
            if child_texts:
                return f"FUNCTION{extra_attrs}({', '.join(child_texts)})"
            return text

        if tag == "operator":
            extra_attrs = self._compact_formula_attrs(attrs, exclude={"name", "id", "key", "label", "ref", "refid", "code", "type"})
            if name and child_texts:
                return f"OPERATOR {name}{extra_attrs}({', '.join(child_texts)})"
            if name:
                return f"OPERATOR {name}{extra_attrs}"
            if child_texts:
                return f"OPERATOR{extra_attrs}({', '.join(child_texts)})"
            return text

        if tag in {"rule_element_ref", "component_ref", "formula_ref", "plan_ref", "measurement_ref", "incentive_ref", "output_reference", "mdltvar_ref"}:
            ref_name = name or text
            extra_attrs = self._compact_formula_attrs(attrs, exclude={"name", "id", "key", "label", "ref", "refid", "code", "type"})
            if ref_name and child_texts:
                suffix = f"{extra_attrs}" if extra_attrs else ""
                return f"{label} {ref_name}{suffix}({', '.join(child_texts)})"
            if ref_name:
                return f"{label} {ref_name}{extra_attrs}"
            if child_texts:
                suffix = f"{extra_attrs}" if extra_attrs else ""
                return f"{label}{suffix}({', '.join(child_texts)})"
            return label

        if tag in {"data_field", "string_literal", "value", "credit_type", "hold_ref", "boolean", "period_type", "relation_type", "event_type"}:
            if text:
                return text
            return name

        if tag == "measurement_ref" or tag == "incentive_ref":
            ref_name = name or text
            extra_attrs = self._compact_formula_attrs(attrs, exclude={"name", "id", "key", "label"})
            if ref_name:
                return f"{label} {ref_name}{extra_attrs}"
            if extra_attrs:
                return f"{label}{extra_attrs}"
            return label

        if tag == "output_reference":
            ref_name = name or text
            extra_attrs = self._compact_formula_attrs(
                attrs,
                exclude={"name", "id", "key", "label", "display_name_for_reports"},
                preferred=("type", "unit_type", "period_type", "display_name_for_reports"),
            )
            if ref_name:
                return f"{label} {ref_name}{extra_attrs}"
            if extra_attrs:
                return f"{label}{extra_attrs}"
            return label

        if tag == "mdltvar_ref":
            ref_name = name or text
            extra_attrs = self._compact_formula_attrs(attrs, exclude={"name", "id", "key", "label", "ref", "refid"})
            if ref_name:
                return f"{label} {ref_name}{extra_attrs}"
            if extra_attrs:
                return f"{label}{extra_attrs}"
            return label

        if name and text and name != text:
            return f"{label} {name}: {text}"
        if name:
            return f"{label} {name}"
        if text:
            return text
        if child_texts:
            return f"{label}({', '.join(child_texts)})"
        return ""

    def _formula_node_tags(self) -> set[str]:
        return {
            "operator",
            "function",
            "rule_element_ref",
            "data_field",
            "string_literal",
            "value",
            "credit_type",
            "output_reference",
            "hold_ref",
            "boolean",
            "period_type",
            "relation_type",
            "event_type",
            "component_ref",
            "formula_ref",
            "plan_ref",
            "measurement_ref",
            "incentive_ref",
            "mdltvar_ref",
        }

    def _compact_formula_attrs(
        self,
        attrs: Dict[str, str],
        exclude: Optional[set[str]] = None,
        preferred: Optional[Iterable[str]] = None,
    ) -> str:
        exclude = exclude or set()
        keys = list(preferred) if preferred else list(attrs.keys())
        parts: List[str] = []
        seen: set[str] = set()
        for key in keys:
            normalized_key = key.lower()
            if normalized_key in exclude or normalized_key in seen:
                continue
            seen.add(normalized_key)
            value = attrs.get(normalized_key)
            if value:
                parts.append(f"{normalized_key.upper()}={value}")
        if not parts:
            return ""
        return " [" + ", ".join(parts) + "]"

    def _build_pseudo_code(
        self,
        record_type: str,
        rule_name: str,
        variables: List[str],
        references: List[str],
        formula: str,
    ) -> str:
        parts: List[str] = []
        if record_type:
            parts.append(f"Type: {record_type}")
        if rule_name:
            parts.append(f"Rule: {rule_name}")
        if variables:
            parts.append(f"Inputs: {', '.join(variables)}")
        if references:
            parts.append(f"References: {', '.join(references)}")
        if formula:
            parts.append(f"Logic: {formula}")
        return " | ".join(parts)

    def _extract_metadata(self, node: ET.Element) -> Dict[str, Any]:
        return {
            "tag": _local_name(node.tag),
            "attributes": _extract_attributes(node),
        }

    def _dedupe_join(self, values: Iterable[str]) -> str:
        result: List[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = _normalize_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return " ".join(result)

    def _extract_name_bs4(self, node: Any) -> str:
        attrs = getattr(node, "attrs", {}) or {}
        lower_attrs = {str(key).lower(): value for key, value in attrs.items()}
        for attr in self.NAME_ATTRS:
            value = _normalize_text(lower_attrs.get(attr.lower()))
            if value:
                return value
        for child in node.find_all(["name", "rulename", "rule_name", "title"]):
            text = _normalize_text(child.get_text(" ", strip=True))
            if text:
                return text
        return ""

    def _extract_description_bs4(self, node: Any) -> str:
        attrs = getattr(node, "attrs", {}) or {}
        lower_attrs = {str(key).lower(): value for key, value in attrs.items()}
        for attr in ("description", "desc", "summary", "comment"):
            value = _normalize_text(lower_attrs.get(attr))
            if value:
                return value
        for child in node.find_all(self.DESCRIPTION_TAGS):
            text = _normalize_text(child.get_text(" ", strip=True))
            if text:
                return text
        return ""

    def _extract_variables_bs4(self, node: Any) -> List[str]:
        variables: List[str] = []
        seen: set[str] = set()
        for child in node.find_all(self.VARIABLE_TAGS):
            candidate = ""
            for attr in ("name", "id", "ref", "refid", "key", "code", "path"):
                candidate = _normalize_text(child.get(attr))
                if candidate:
                    break
            if not candidate:
                candidate = _normalize_text(child.get_text(" ", strip=True))
            if candidate and candidate not in seen:
                seen.add(candidate)
                variables.append(candidate)
        return variables

    def _extract_references_bs4(self, node: Any) -> List[str]:
        references: List[str] = []
        seen: set[str] = set()
        for child in node.find_all(["rule_element_ref", "component_ref", "formula_ref", "plan_ref"]):
            candidate = ""
            for attr in ("name", "id", "key", "label", "rule", "rule_name"):
                candidate = _normalize_text(child.get(attr))
                if candidate:
                    break
            if candidate and candidate not in seen:
                seen.add(candidate)
                references.append(candidate)
        return references

    def _extract_formula_bs4(self, node: Any) -> str:
        parts: List[str] = []
        search_tags = list(self.EXPRESSION_TAGS) + [
            "operator",
            "function",
            "rule_element_ref",
            "data_field",
            "string_literal",
            "value",
            "credit_type",
            "output_reference",
            "hold_ref",
            "boolean",
            "period_type",
            "relation_type",
            "component_ref",
        ]
        for child in node.find_all(True):
            if _local_name(getattr(child, "name", "")) not in set(search_tags):
                continue
            text = _normalize_text(child.get_text(" ", strip=True))
            if text:
                parts.append(text)
        if parts:
            return self._dedupe_join(parts)
        return _normalize_text(node.get_text(" ", strip=True))


def parse_sap_xml(xml_content: str) -> List[Dict[str, Any]]:
    """Convenience wrapper for one-off parsing."""

    return SAPXMLParser().parse_string(xml_content)


def parse_sap_xml_file(file_path: str | Path) -> List[Dict[str, Any]]:
    """Convenience wrapper for file-based parsing."""

    return SAPXMLParser().parse_file(file_path)


def export_parsed_rules_to_json(xml_content: str) -> str:
    """Serialize parsed rules into pretty JSON for inspection or storage."""

    return json.dumps(parse_sap_xml(xml_content), indent=2, ensure_ascii=False)
