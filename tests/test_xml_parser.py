from app.data_pipeline.xml_parser import SAPXMLParser


def test_parse_string_extracts_rule_name_description_variables_and_formula():
    xml = """
    <DATA_IMPORT>
      <RULE_SET>
        <RULE NAME="Rule_A" DESCRIPTION="Sample rule">
          <ACTION_EXPRESSION_SET>
            <ACTION_EXPRESSION>
              <FUNCTION ID="ifThenElse">
                <DATA_FIELD>SalesTransaction.value</DATA_FIELD>
                <STRING_LITERAL>100</STRING_LITERAL>
              </FUNCTION>
            </ACTION_EXPRESSION>
          </ACTION_EXPRESSION_SET>
          <CONDITION_EXPRESSION>
            <OPERATOR ID="ISEQUALTO_OPERATOR">
              <DATA_FIELD>Position.status</DATA_FIELD>
              <STRING_LITERAL>ACTIVE</STRING_LITERAL>
            </OPERATOR>
          </CONDITION_EXPRESSION>
        </RULE>
      </RULE_SET>
    </DATA_IMPORT>
    """

    parser = SAPXMLParser()
    records = parser.parse_string(xml)

    assert len(records) == 1
    record = records[0]
    assert record["record_type"] == "rule"
    assert record["rule_name"] == "Rule_A"
    assert record["description"] == "Sample rule"
    assert "SalesTransaction.value" in record["formula"]
    assert "Position.status" in record["formula"]
    assert "Type: rule" in record["pseudo_code"]


def test_parse_string_falls_back_to_bs4_for_malformed_xml():
    xml = "<RULE NAME='Rule_B'><DESCRIPTION>Broken XML"

    parser = SAPXMLParser()
    records = parser.parse_string(xml)

    assert records
    assert records[0]["record_type"] == "rule"


def test_parse_nested_function_and_rule_element_ref_is_structured():
    xml = """
    <RULE NAME="Rule_C" DESCRIPTION="Nested example">
      <ACTION_EXPRESSION_SET>
        <ACTION_EXPRESSION>
          <FUNCTION ID="ifThenElse">
            <DATA_FIELD>SalesTransaction.value</DATA_FIELD>
            <STRING_LITERAL>100</STRING_LITERAL>
            <RULE_ELEMENT_REF NAME="HoldRule" />
          </FUNCTION>
        </ACTION_EXPRESSION>
      </ACTION_EXPRESSION_SET>
      <CONDITION_EXPRESSION>
        <OPERATOR ID="ISEQUALTO_OPERATOR">
          <DATA_FIELD>Position.status</DATA_FIELD>
          <STRING_LITERAL>ACTIVE</STRING_LITERAL>
        </OPERATOR>
      </CONDITION_EXPRESSION>
    </RULE>
    """

    parser = SAPXMLParser()
    records = parser.parse_string(xml)

    assert len(records) == 1
    record = records[0]
    assert "FUNCTION ifThenElse" in record["formula"]
    assert "RULE_ELEMENT_REF HoldRule" in record["formula"]
    assert "SalesTransaction.value" in record["variables"]
    assert "Position.status" in record["variables"]
    assert "HoldRule" in record["references"]


def test_parse_realistic_nested_sap_formula_nodes_preserves_structure():
    xml = """
    <FORMULA NAME="F_Direct_Credit_Before_Termination_Check" DESCRIPTION="Formula used for crediting till termination date">
      <FORMULA_EXPRESSION>
        <OPERATOR ID="OR_OPERATOR">
          <OPERATOR ID="LESSTHANEQUALTO_OPERATOR" PAREN_WRAPPED="true">
            <DATA_FIELD>SalesTransaction.compensationDate</DATA_FIELD>
            <DATA_FIELD>Position.genericDate3</DATA_FIELD>
          </OPERATOR>
          <FUNCTION ID="isDateNull" PAREN_WRAPPED="true">
            <DATA_FIELD>Position.genericDate3</DATA_FIELD>
          </FUNCTION>
        </OPERATOR>
      </FORMULA_EXPRESSION>
    </FORMULA>
    """

    parser = SAPXMLParser()
    records = parser.parse_string(xml)

    assert len(records) == 1
    record = records[0]
    assert "OPERATOR OR_OPERATOR" in record["formula"]
    assert "OPERATOR LESSTHANEQUALTO_OPERATOR [PAREN_WRAPPED=true]" in record["formula"]
    assert "FUNCTION isDateNull [PAREN_WRAPPED=true]" in record["formula"]
    assert "SalesTransaction.compensationDate" in record["variables"]
    assert "Position.genericDate3" in record["variables"]


def test_parse_realistic_rule_output_reference_and_mdltvar_ref():
    xml = """
    <RULE NAME="CR_O_KPI_Initial_Premium_Direct" DESCRIPTION="Direct credit rule for Initial Premium KPI">
      <ACTION_EXPRESSION_SET>
        <ACTION_EXPRESSION>
          <FUNCTION ID="DIRECT_TRANSACTION_CREDIT_ALLGAs">
            <OUTPUT_REFERENCE NAME="C_O_KPI_Initial_Premium_Direct" TYPE="Credit" UNIT_TYPE="USD" />
            <OPERATOR ID="MULTIPLY_OPERATOR">
              <DATA_FIELD>SalesTransaction.value</DATA_FIELD>
              <FUNCTION ID="NULL_TO_VALUE">
                <FUNCTION ID="MDLT_FUNCTION">
                  <MDLTVAR_REF NAME="LTV_Product_Adjustment_Ratio" RETURN_TYPE="Quantity" />
                  <DATA_FIELD>SalesTransaction.genericDate3</DATA_FIELD>
                  <DATA_FIELD>SalesTransaction.productId</DATA_FIELD>
                  <STRING_LITERAL><![CDATA[IP]]></STRING_LITERAL>
                </FUNCTION>
                <VALUE DECIMAL_VALUE="1" UNIT_TYPE="quantity" />
              </FUNCTION>
            </OPERATOR>
          </FUNCTION>
        </ACTION_EXPRESSION>
      </ACTION_EXPRESSION_SET>
    </RULE>
    """

    parser = SAPXMLParser()
    records = parser.parse_string(xml)

    assert len(records) == 1
    record = records[0]
    assert "FUNCTION DIRECT_TRANSACTION_CREDIT_ALLGAs" in record["formula"]
    assert "OUTPUT_REFERENCE C_O_KPI_Initial_Premium_Direct" in record["formula"]
    assert "MDLTVAR_REF LTV_Product_Adjustment_Ratio" in record["formula"]
    assert "C_O_KPI_Initial_Premium_Direct" in record["references"]
    assert "LTV_Product_Adjustment_Ratio" in record["references"]
