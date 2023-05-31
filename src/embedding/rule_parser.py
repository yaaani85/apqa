from rule import Rule
from literal import Literal

class RuleParser:
    def __init__(self):
        pass

    def parse_literal(self, literal_str):
        start_terms = literal_str.index('(')
        predicate_name = literal_str[:start_terms]
        is_negated = False
        if predicate_name.startswith('~'):
            predicate_name = predicate_name[1:]
            is_negated = True
        end_terms = literal_str.index(')')
        terms_str = literal_str[start_terms + 1 : end_terms]
        terms = terms_str.split(',')
        rest = literal_str[end_terms + 1:]
        return Literal(predicate_name, terms, is_negated=is_negated), rest

    def parse_conjunction_literals(self, cnj_str):
        out = []
        while cnj_str is not None and cnj_str != '':
            cnj_str = cnj_str.strip()
            literal, cnj_str = self.parse_literal(cnj_str)
            cnj_str = cnj_str.strip()
            if cnj_str.startswith(','):
                cnj_str = cnj_str[1:]
            out.append(literal)
        return out

    def parse_rule(self, rule_str):
        head, body = rule_str.split(':-')
        head = head.strip()
        body = body.strip()
        head_literals = self.parse_conjunction_literals(head)
        body_literals = self.parse_conjunction_literals(body)
        return Rule(head_literals, body_literals)
    


    def parse_query(self, datalog_query):
        parser = RuleParser()
        parsed_rule = parser.parse_rule(datalog_query)
        parsed_rule_head = parsed_rule.get_head()
        rule_body = parsed_rule.get_body()
        parsed_rule_body = [tuple(literal.get_terms()) for literal in rule_body]

        return parsed_rule_head, parsed_rule_body
            
  