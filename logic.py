"""
Representations and Inference for Logic. (Chapters 7-9, 12)

Covers both Propositional and First-Order Logic. First we have four
important data types:

    KB            Abstract class holds a knowledge base of logical expressions
    KB_Agent      Abstract class subclasses agents.Agent
    Expr          A logical expression, imported from utils.py
    substitution  Implemented as a dictionary of var:value pairs, {x:1, y:x}

Be careful: some functions take an Expr as argument, and some take a KB.

Logical expressions can be created with Expr or expr, imported from utils, TODO
or with expr, which adds the capability to write a string that uses
the connectives ==>, <==, <=>, or <=/=>. But be careful: these have the
operator precedence of commas; you may need to add parens to make precedence work.
See logic.ipynb for examples.

Then we implement various functions for doing logical inference:

    pl_true          Evaluate a propositional logical sentence in a model
    tt_entails       Say if a statement is entailed by a KB
    pl_resolution    Do resolution on propositional sentences
    dpll_satisfiable See if a propositional sentence is satisfiable
    WalkSAT          Try to find a solution for a set of clauses

And a few other functions:

    to_cnf           Convert to conjunctive normal form
    unify            Do unification of two FOL sentences
    diff, simp       Symbolic differentiation and simplification
"""

import heapq
import itertools
import random
from collections import defaultdict, Counter
import networkx as nx
from utils import *


class KB:
    """A knowledge base to which you can tell and ask sentences.
    To create a KB, first subclass this class and implement
    tell, ask_generator, and retract. Why ask_generator instead of ask?
    The book is a bit vague on what ask means --
    For a Propositional Logic KB, ask(P & Q) returns True or False, but for an
    FOL KB, something like ask(Brother(x, y)) might return many substitutions
    such as {x: Cain, y: Abel}, {x: Abel, y: Cain}, {x: George, y: Jeb}, etc.
    So ask_generator generates these one at a time, and ask either returns the
    first one or returns False."""

    def __init__(self, sentence=None):
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        """Add the sentence to the KB."""
        raise NotImplementedError

    def ask(self, query):
        """Return a substitution that makes the query true, or, failing that, return False."""
        return first(self.ask_generator(query), default=False)

    def ask_generator(self, query):
        """Yield all the substitutions that make query true."""
        raise NotImplementedError

    def retract(self, sentence):
        """Remove sentence from the KB."""
        raise NotImplementedError

# Generic Knowledge Base for Propositional Logic
class PropKB(KB):
    """A KB for propositional logic. Inefficient, with no indexing.
    This is a simple KB that can be used for propositional logic. (the generic type of KB)"""

    def __init__(self, sentence=None):
        super().__init__(sentence)
        self.clauses = []

    def tell(self, sentence):
        self.clauses.extend(conjuncts(to_cnf(sentence)))      

    def ask_generator(self, query):
        """Yield the empty substitution {} if KB entails query; else no results."""
        return tt_entails(Expr('&', *self.clauses), query)
             

    def ask_if_true(self, query):
        """Return True if the KB entails query, else return False."""
        for _ in self.ask_generator(query):
            return True
        return False

    def retract(self, sentence):
        """Remove the sentence's clauses from the KB."""
        for c in conjuncts(to_cnf(sentence)):
            if c in self.clauses:
                self.clauses.remove(c)

# Propositional Logic Knowledge Base of definite clauses
class PropDefiniteKB(PropKB):
    """A KB of propositional definite clauses.
    A definite clause is a specific clause type of horn clause"""

    def tell(self, sentence):
        """Add a definite clause to this KB."""
        assert is_definite_clause(sentence), "Must be definite clause"
        self.clauses.append(sentence)

    def ask_generator(self, query):
        """Yield the empty substitution if KB implies query; else nothing."""
        if pl_fc_entails(self.clauses, query):
            yield {}

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def clauses_with_premise(self, p):
        """Return a list of the clauses in KB that have p in their premise.
        This could be cached away for O(1) speed, but we'll recompute it."""
        return [c for c in self.clauses if c.op == '==>' and p in conjuncts(c.args[0])]

# Propositional Logic Knowledge Base of horn clauses
class HornClauseKB(PropKB):
    """A KB of propositional horn clauses.
    A horn clause is a the clause with at most one positive literal."""

    def tell(self, sentence):
        """Add a horn clause to this KB."""
        self.clauses.append(sentence)

    def ask_generator(self, query):
        """Yield the empty substitution if KB implies query; else nothing."""
        if pl_fc_entails(self.clauses, query):
            yield {}
        
    def retract(self, sentence):
        self.clauses.remove(sentence)


    def clauses_with_premise(self, p):
        """Return a list of the clauses in KB that have p in their premise.
        This could be cached away for O(1) speed, but we'll recompute it."""
        return [c for c in self.clauses if c.op == '==>' and p in conjuncts(c.args[0])]

    def clauses_with_consequent(self, symbol):
        # Return clauses where symbol is the consequent
        return [c for c in self.clauses if c.op == '==>' and c.args[1] == symbol]


def is_horn_clause(s):
    """return True for exprs s of the form horn clause, 
    That is, at most one positive literal.
    """
    if is_symbol(s.op):
        return True
    elif s.op == '~':
        return is_symbol(s.args[0].op) #detect single negative literal 
    elif s.op == '==>':
        antecedent, consequent = s.args
        # Allow consequent to be a symbol (definite clause) or False (negative clause)
        if consequent == False or is_symbol(consequent.op):
            # Ensure all conjuncts in the antecedent are symbols
            return all(is_symbol(arg.op) for arg in conjuncts(antecedent))
        return False
    elif s.op == '|':
        # Check disjunctive form directly: at most one positive literal
        positive_literals = [lit for lit in disjuncts(s) if lit.op != '~']
        return len(positive_literals) <= 1
    else:
        return False


def parse_horn_clause(s):
    """Return the antecedents and the consequent of a horn clause."""
    assert is_horn_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent


def is_definite_clause(s):
    """Returns True for exprs s of the form A & B & ... & C ==> D,
    where all literals are positive. In clause form, this is
    ~A | ~B | ... | ~C | D, where exactly one clause is positive.
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    """
    if is_symbol(s.op):
        return True
    elif s.op == '==>':
        antecedent, consequent = s.args
        return is_symbol(consequent.op) and all(is_symbol(arg.op) for arg in conjuncts(antecedent))
        # return is_symbol(consequent.op) and all(is_symbol(arg.op) for arg in conjuncts(antecedent))
    else:
        return False


def parse_definite_clause(s):
    """Return the antecedents and the consequent of a definite clause."""
    assert is_definite_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent


def parse_logical_expr(s):
    """
    Parse a string into an Expr, handling infix => and <=>.
    Example: 'a <=> (b => c)' => Expr('<=>', a, Expr('=>', b, c))
    """
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    tokens = s.split()
    return parse_expr_tokens(tokens)


def parse_expr_tokens(tokens):
    """
    Recursive parser from token list → Expr
    """
    def parse_expr():
        token = next_token()
        if token == '(':
            sub_expr = parse_equiv()
            assert next_token() == ')'
            return sub_expr
        elif token == '~':
            return Expr('~', parse_expr())
        else:
            return expr(token)

    def parse_equiv():
        left = parse_implies()
        while peek() == '<=>':
            next_token()
            right = parse_implies()
            left = Expr('<=>', left, right)
        return left

    def parse_implies():
        left = parse_or()
        while peek() == '=>':
            next_token()
            right = parse_or()
            left = Expr('==>', left, right)
        return left

    def parse_or():
        left = parse_and()
        while peek() == '|':
            next_token()
            right = parse_and()
            left = Expr('|', left, right)
        return left

    def parse_and():
        left = parse_expr()
        while peek() == '&':
            next_token()
            right = parse_expr()
            left = Expr('&', left, right)
        return left

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def next_token():
        token = tokens[pos[0]]
        pos[0] += 1
        return token

    pos = [0]
    return parse_equiv()


# ______________________________________________________________________________
# Truth Table Logic
def tt_entails(kb, alpha):
    """
    Does kb entail the sentence alpha? Use truth tables. For propositional
    kb's and sentences. Note that the 'kb' should be an Expr which is a
    conjunction of clauses.
    >>> tt_entails(expr('P & Q'), expr('Q'))
    True
    """
    symbols = list(prop_symbols(kb & alpha))
    model_count = 0
    result, model_count = tt_check_all(kb, alpha, symbols, {}, model_count)
    return result, model_count


def tt_check_all(kb, alpha, symbols, model, model_count):
    if not symbols:
        kb_val = pl_true(kb, model)
        if kb_val is True:
            # KB is true in this model → alpha must be True
            if pl_true(alpha, model):
                model_count += 1
                return True, model_count
            else:
                return False, model_count  # Fail if alpha is False in a KB-True model
        else:
            # KB is False or Undecided — this model doesn't affect entailment
            return True, model_count
    else:
        P, rest = symbols[0], symbols[1:]
        
        # Evaluate with P=True
        model_true = extend(model, P, True)
        true_result, true_count = tt_check_all(kb, alpha, rest, model_true, model_count)

        # Evaluate with P=False
        model_false = extend(model, P, False)
        false_result, false_count = tt_check_all(kb, alpha, rest, model_false, true_count)

        # For entailment, both branches must return True
        return true_result and false_result, false_count


def pl_true(exp, model={}):
    """Return True if the propositional logic expression is true in the model,
    and False if it is false. If the model does not specify the value for
    every proposition, this may return None to indicate 'not obvious';
    this may happen even when the expression is tautological.
    >>> pl_true(P, {}) is None
    True
    """
    if exp in (True, False):
        return exp
    op, args = exp.op, exp.args
    if is_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None:
            return None
        else:
            return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True:
                return True
            if p is None:
                result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False:
                return False
            if p is None:
                result = None
        return result
     # Handle binary operators
    p, q = args
    if op == '==>':
        return pl_true(~p | q, model)
    elif op == '<==':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None:
        return None
    qt = pl_true(q, model)
    if qt is None:
        return None
    if op == '<=>':
        return pt == qt
    elif op == '^':  # xor or 'not equivalent'
        return pt != qt
    else:
        raise ValueError('Illegal operator in logic expression' + str(exp))


# ______________________________________________________________________________
# Convert to Conjunctive Normal Form (CNF)
def to_cnf(s):
    """
    [Page 253]
    Convert a propositional logical sentence to conjunctive normal form.
    That is, to the form ((A | ~B | ...) & (B | C | ...) & ...)
    >>> to_cnf('~(B | C)')
    (~B & ~C)
    """
    s = expr(s)
    if isinstance(s, str):
        s = expr(s)
    s = eliminate_implications(s)  # Steps 1, 2 from p. 253
    s = move_not_inwards(s)  # Step 3
    return distribute_and_over_or(s)  # Step 4


def eliminate_implications(s):
    """Change implications into equivalent form with only &, |, and ~ as logical operators."""
    s = expr(s)
    if not s.args or is_symbol(s.op):
        return s  # Atoms are unchanged.
    args = list(map(eliminate_implications, s.args))
    a, b = args[0], args[-1]
    if s.op == '==>':
        return b | ~a
    elif s.op == '<==':
        return a | ~b
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    elif s.op == '^':
        assert len(args) == 2  # TODO: relax this restriction
        return (a & ~b) | (~a & b)
    else:
        assert s.op in ('&', '|', '~')
        return Expr(s.op, *args)


def move_not_inwards(s):
    """Rewrite sentence s by moving negation sign inward.
    >>> move_not_inwards(~(A | B))
    (~A & ~B)
    """
    s = expr(s)
    if s.op == '~':
        def NOT(b):
            return move_not_inwards(~b)

        a = s.args[0]
        if a.op == '~':
            return move_not_inwards(a.args[0])  # ~~A ==> A
        if a.op == '&':
            return associate('|', list(map(NOT, a.args)))
        if a.op == '|':
            return associate('&', list(map(NOT, a.args)))
        return s
    elif is_symbol(s.op) or not s.args:
        return s
    else:
        return Expr(s.op, *list(map(move_not_inwards, s.args)))


def distribute_and_over_or(s):
    """Given a sentence s consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in CNF.
    >>> distribute_and_over_or((A & B) | C)
    ((A | C) & (B | C))
    """
    s = expr(s)
    if s.op == '|':
        s = associate('|', s.args)
        if s.op != '|':
            return distribute_and_over_or(s)
        if len(s.args) == 0:
            return False
        if len(s.args) == 1:
            return distribute_and_over_or(s.args[0])
        conj = first(arg for arg in s.args if arg.op == '&')
        if not conj:
            return s
        others = [a for a in s.args if a is not conj]
        rest = associate('|', others)
        return associate('&', [distribute_and_over_or(c | rest)
                               for c in conj.args])
    elif s.op == '&':
        return associate('&', list(map(distribute_and_over_or, s.args)))
    else:
        return s


def associate(op, args):
    """Given an associative op, return an expression with the same
    meaning as Expr(op, *args), but flattened -- that is, with nested
    instances of the same op promoted to the top level.
    >>> associate('&', [(A&B),(B|C),(B&C)])
    (A & B & (B | C) & B & C)
    >>> associate('|', [A|(B|(C|(A&B)))])
    (A | B | C | (A & B))
    """
    args = dissociate(op, args)
    if len(args) == 0:
        return _op_identity[op]
    elif len(args) == 1:
        return args[0]
    else:
        return Expr(op, *args)


_op_identity = {'&': True, '|': False, '+': 0, '*': 1}


def dissociate(op, args):
    """Given an associative op, return a flattened list result such
    that Expr(op, *result) means the same as Expr(op, *args).
    >>> dissociate('&', [A & B])
    [A, B]
    """
    result = []

    def collect(subargs):
        for arg in subargs:
            if arg.op == op:
                collect(arg.args)
            else:
                result.append(arg)

    collect(args)
    return result


def conjuncts(s):
    """Return a list of the conjuncts in the sentence s.
    >>> conjuncts(A & B)
    [A, B]
    >>> conjuncts(A | B)
    [(A | B)]
    """
    return dissociate('&', [s])


def disjuncts(s):
    """Return a list of the disjuncts in the sentence s.
    >>> disjuncts(A | B)
    [A, B]
    >>> disjuncts(A & B)
    [(A & B)]
    """
    return dissociate('|', [s])


# ______________________________________________________________________________
# Propositional Logic Forward and Backward Chaining
def pl_fc_entails(kb, q):
    """
    Use forward chaining to see if a PropDefiniteKB entails symbol q.
    >>> pl_fc_entails(horn_clauses_KB, expr('Q'))
    True
    """
    # count[c] is initially the number of symbols in clause c’s premise
    count = {c: len(conjuncts(c.args[0])) for c in kb.clauses if c.op == '==>'}

    # inferred[s] is initially false for all symbols
    # Prevents reprocessing symbols already inferred.
    inferred = defaultdict(bool)

    # a queue of symbols, initially symbols known to be true in KB
    agenda = [s for s in kb.clauses if is_symbol(s.op)]

    #list of entailed symbols: using this to return the entailed symbols for printing output
    entailed_symbols = []

    while agenda:
        p = agenda.pop()
        if p == q:
            entailed_symbols.append(p)
            return True, [str(s) for s in entailed_symbols]
        #loop through all clauses that contains p
        if not inferred[p]:
            inferred[p] = True
            entailed_symbols.append(p)
            for c in kb.clauses_with_premise(p):
                count[c] -= 1
                #when all premises are true => we can infer the consequent
                if count[c] == 0:
                    agenda.append(c.args[1])
    return False, []


def pl_bc_entails(kb, q):
    """
    Use backward chaining to see if a PropDefiniteKB entails symbol q.
    >>> pl_fc_entails(horn_clauses_KB, expr('Q'))
    True
    """

    def bc_prove(symbol, kb, inferred, used_symbols, proof_trace):
        """Recursive helper function for backward chaining."""
        # Check if symbol is already proved to avoid redundant work
        if inferred[symbol]:
            return True
        
        # Prevent infinite recursion by checking for cycles
        if symbol in used_symbols:
            return False
        
        used_symbols.add(symbol)
                         
        # If symbol is a fact in KB, mark as proved and return True
        if any(symbol == fact for fact in kb.clauses if is_symbol(fact.op)):
            inferred[symbol] = True
            proof_trace.append(symbol)
            return True

        # Check all clauses in KB that have symbol as a consequent
        for clause in kb.clauses_with_consequent(symbol):
            premises = conjuncts(clause.args[0])
            if all(bc_prove(premise, kb, inferred, used_symbols.copy(), proof_trace) for premise in premises):
                inferred[symbol] = True
                proof_trace.append(symbol)
                return True
        # If no clauses lead to symbol, return False
        return False 

    # Initialize dictionary to track proved symbols (False by default)
    inferred = defaultdict(bool)
    # Initialize set to track symbols in current proof path for cycle detection
    used_symbols = set()
    # Initialize proof trace to keep track of the symbols used in the proof
    proof_trace = []
    # Start recursive proof with query symbol
    result = bc_prove(q, kb, inferred, used_symbols, proof_trace)

    # Return (True, symbol list) if proved, (False, []) if not
    return (result, [str(s) for s in proof_trace] if result else [])


# ______________________________________________________________________________
# Propositional Logic Resolution 
def pl_resolution(kb, alpha):
    """
    Propositional-logic resolution: say if alpha follows from KB.
    >>> pl_resolution(horn_clauses_KB, A)
    True
    """
    clauses = kb.clauses + conjuncts(to_cnf(~alpha))
    new = set()
    count = 0

    while True:
        n = len(clauses)
        pairs = [(clauses[i], clauses[j])
                 for i in range(n) for j in range(i + 1, n)]
        for (ci, cj) in pairs:   
            resolvents = pl_resolve(ci, cj)
            # Increment by number of resolvents generated
            count += len(resolvents)
            if False in resolvents:
                return True, count
            new = new.union(set(resolvents))
        if new.issubset(set(clauses)):
            return False, count
        for c in new:
            if c not in clauses:
                clauses.append(c)

def pl_resolve(ci, cj):
    """Return all clauses that can be obtained by resolving clauses ci and cj."""
    clauses = []
    for di in disjuncts(ci):
        for dj in disjuncts(cj):
            if di == ~dj or ~di == dj:
                clauses.append(associate('|', unique(remove_all(di, disjuncts(ci)) + remove_all(dj, disjuncts(cj)))))
    return clauses


# ______________________________________________________________________________
# Propositional Logic DPLL 
def dpll_entails(kb_clauses, query):
    """
    Return True if KB ⊨ query using DPLL.
    This is done by checking if KB ∧ ¬query is unsatisfiable.
    """
    combined = Expr('&', *kb_clauses, ~query)          # KB ∧ ¬query
    cnf_clauses = conjuncts(to_cnf(combined))          # Convert to CNF
    symbols = list(prop_symbols(combined))             # All propositional symbols
    result = dpll(cnf_clauses, symbols, {})            # Run DPLL
    return result is False                             # UNSAT → Entailment holds

def dpll(clauses, symbols, model):
    """Core DPLL satisfiability checking logic."""
    if all(pl_true(c, model) for c in clauses):
        return model  # All true → SAT
    if any(pl_true(c, model) is False for c in clauses):
        return False  # Some false → UNSAT

    # Pure symbol
    pure, value = find_pure_symbol(symbols, clauses, model)
    if pure:
        return dpll(clauses, [s for s in symbols if s != pure], extend(model, pure, value))

    # Unit clause
    unit, value = find_unit_clause(clauses, model)
    if unit:
        return dpll(clauses, [s for s in symbols if s != unit], extend(model, unit, value))

    # Try remaining symbol
    if symbols:
        P, rest = symbols[0], symbols[1:]
        return (dpll(clauses, rest, extend(model, P, True)) or
                dpll(clauses, rest, extend(model, P, False)))

    return False  # No symbols and some clauses unknown → UNSAT

def find_pure_symbol(symbols, clauses, model):
    """Return (symbol, value) if a pure symbol is found."""
    for s in symbols:
        if s in model:
            continue
        found_pos = found_neg = False
        for clause in clauses:
            if pl_true(clause, model) is not None:
                continue
            literals = disjuncts(clause)
            if s in literals:
                found_pos = True
            if Expr('~', s) in literals:
                found_neg = True
        if found_pos != found_neg:
            return s, found_pos
    return None, None

def find_unit_clause(clauses, model):
    """Return (symbol, value) if a unit clause is found."""
    for clause in clauses:
        if pl_true(clause, model) is not None:
            continue
        literals = disjuncts(clause)
        unassigned = None
        count = 0
        for lit in literals:
            val = pl_true(lit, model)
            if val is None:
                unassigned = lit
                count += 1
        if count == 1:
            return (
                unassigned.args[0] if unassigned.op == '~' else unassigned,
                False if unassigned.op == '~' else True
            )
    return None, None

