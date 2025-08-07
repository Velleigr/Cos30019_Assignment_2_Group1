import sys
from utils import *
from logic import tt_entails, is_horn_clause, HornClauseKB, PropKB, parse_logical_expr, conjuncts, to_cnf, pl_fc_entails, pl_bc_entails, tt_entails, pl_resolution, dpll_entails

conventional_symbols ={
    '^': '&', '∧': '&',
    '¬':  '~',   '!': '~',
    '->': '==>',  '→': '==>',
    '<-': '<==',  '←': '<==',
    '<->':'<=>', '↔': '<=>', 
    '||': '|', 
}

def prepare_kb_for_custom_generic(kb):
    """
    Given a KB (HornClauseKB or PropKB), convert all clauses to CNF and return a PropKB.
    This ensures the KB is ready for Resolution or DPLL. 
    """
    kb_clauses = []
    for clause in kb.clauses:
        cnf_clauses = conjuncts(to_cnf(clause))
        kb_clauses.extend(cnf_clauses)

    kb_for_custom = PropKB()
    kb_for_custom.clauses = kb_clauses

    return kb_for_custom

def parse_input_file(filename):
    kb = None
    query = None
    clauses = []

    with open(filename, 'r') as file:
        data = file.readlines()

    # Process the input file line by line
    # check if we are in the query section
    in_query_section = False 

    # Iterate through each line in the file
    for line in data:
        line = line.strip()
        if not line:  # skip blank line
            continue
        if line.lower() == "tell":  # skip TELL
            continue
        if line.lower() == "ask":  # find ASK
            in_query_section = True
            continue
        
        if in_query_section:
            query = parse_logical_expr(line.strip())  # parse the query
        else:
            # get the knowledge base
            for sub_line in line.split(';'):
                sub_line = sub_line.strip()
                # skip empty sub_lines
                if not sub_line:
                    continue
                # replace conventional symbols with logical operators
                for symbol, replacement in conventional_symbols.items():
                    sub_line = sub_line.replace(symbol, replacement)

                clauses.append(sub_line)

    if not clauses:
        print("Error: No knowledge base (TELL section is empty).")
        sys.exit(1)

    if not query:
        print("Error: No query found (ASK section is missing).")
        sys.exit(1)

    expr_clauses = [parse_logical_expr(clause) for clause in clauses]
    # expr_clauses = [expr(expr_handle_infix_imp(expr_handle_infix_or(clause))) for clause in clauses]
    all_horn = all(is_horn_clause(clause_expr) for clause_expr in expr_clauses)

    kb = HornClauseKB() if all_horn else PropKB()

    for clause_expr in expr_clauses:
        kb.tell(clause_expr)
                   
    return kb, query

def main():
    if len(sys.argv) != 3:
        print("Usage: python iengine.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    if method not in ["TT", "FC", "BC", "RES", "DPLL"]:
        print("Error: Unknown method. Use one of TT, FC, or BC.")
        sys.exit(1)

    kb, query = parse_input_file(filename)

    if (method in ["FC", "BC"]) and not isinstance(kb, HornClauseKB):
        print(f"Error: {method} method requires a Horn Clause KB, but a general KB (PropKB) was provided.")
        sys.exit(1)

    if method == "TT":
        result, model_count = tt_entails(Expr('&', *kb.clauses), expr(query))
        print(f"YES: {model_count}" if result else "NO")
    elif method == "FC":
        result, entailed = pl_fc_entails(kb, expr(query))
        if result:
            print(f"YES: {', '.join(entailed)}")
        else:
            print("NO")
    elif method == "BC":
        result, entailed = pl_bc_entails(kb, expr(query))
        if result:
            print(f"YES: {', '.join(entailed)}")
        else:
            print("NO")
    elif method == "RES":
        result, entailed = pl_resolution(kb, expr(query))
        if result:
            print(f"YES: Count number of pairs of clauses used in resolution: {entailed}")
        else:
            print("NO")
    elif method == "DPLL":
        kb = prepare_kb_for_custom_generic(kb)
        result = dpll_entails(kb.clauses, expr(query))
        if result:
            print("Yes")
        else:
            print("NO")
    else:
        print("Unknown method. Use TT, FC, or BC.")

if __name__ == "__main__":
    main()