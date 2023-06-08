""" SCALLOP/SDD COPYRIGHT"""
import functools
import numpy as np
from pysdd import sdd
from pysdd.iterator import SddIterator
from pysdd.sdd import SddManager, Vtree
from .semiring import SemiringGradient

class WMCSolver():
    def __init__(self, answers, probabilties):
        self.answers = answers
        self.probabilties = probabilties
    


    def gen_weights_from_raw(self, raw_weights, variable_dict, semiring):
        weights = [0.0]

        for idx, _ in enumerate(variable_dict):
            weights.append(raw_weights[idx])
        weights = {fact_id: (semiring.pos_value(w, fact_id-1), semiring.neg_value(w, fact_id-1))
                    for fact_id, w in enumerate(weights) if not fact_id == 0}

        return weights
    # @timeout_decorator.timeout(60)
    def get_wmcs(self):
        wmc_value = {}
        wmc_idxes = {}

        answer, variables_per_answer, contains_deteremnistic_answer = self.answers
        counter = 0
        for oid, proofs in answer.items():
            

            if any(contains_deteremnistic_answer[oid]):
                wmc_value[oid] =  (1.0, np.array([]))
                wmc_idxes[oid] = []
                continue
            probabilistic_variables = [i for i in variables_per_answer[oid] if self.probabilties.get(i) is not None ]
            variable_dict = {org_id: new_id + 1 for new_id, org_id in enumerate(probabilistic_variables)}
            raw_weights = [self.probabilties[idx] for idx in probabilistic_variables ]
            var_count = len(variable_dict)  

            semiring = SemiringGradient(var_count)
            manager = SddManager(var_count=var_count, auto_gc_and_minimize=False)

            weights = self.gen_weights_from_raw(raw_weights, variable_dict, semiring)
            clauses = []

            for fact_id_list in proofs:
                fact_literals = []
                for fact in fact_id_list:
                    try:
                        literal = manager.literal(variable_dict[fact])
                    except KeyError:
                        continue
                    fact_literals.append(literal)

    
                clause = functools.reduce(lambda a, b: a & b, fact_literals)
                clauses.append(clause)

            if len(clauses) == 0:
                continue

            if len(clauses) == 1:
                fact_clauses = clauses[0]

            elif len(clauses) > 1:
                fact_clauses = functools.reduce(lambda a, b: a | b, clauses)

            clauses = fact_clauses
            wmc = self.get_wmc(
                manager,
                clauses,
                weights,
                semiring,
                self.get_wmc_func(
                    weights=weights,
                    semiring=semiring,
                    perform_smoothing=True))

            wmc_value[oid] = semiring.result(wmc)
            wmc_idxes[oid] = list(variable_dict.keys())
            counter += 1
        return wmc_value, wmc_idxes

    def get_wmc_func(self, weights, semiring, perform_smoothing=True):
        """
        Get the function used to perform weighted model counting with the SddIterator. Smoothing supported.

        :param weights: The weights used during computations.
        :type weights: dict[int, tuple[Any, Any]]
        :param semiring: The semiring used for the operations.
        :param perform_smoothing: Whether smoothing must be performed. If false but semiring.is_nsp() then
            smoothing is still performed.
        :return: A WMC function that uses the semiring operations and weights, Performs smoothing if needed.
        """

        smooth_flag = perform_smoothing or semiring.is_nsp()

        def func_weightedmodelcounting(
            node, rvalues, expected_prime_vars, expected_sub_vars
        ):
            """ Method to pass on to SddIterator's ``depth_first`` to perform weighted model counting."""
            if rvalues is None:
                # Leaf
                if node.is_true():
                    result_weight = semiring.one()

                    # If smoothing, go over literals missed in scope
                    if smooth_flag:
                        missing_literals = (
                            expected_prime_vars
                            if expected_prime_vars is not None
                            else set()
                        )
                        missing_literals |= (
                            expected_sub_vars
                            if expected_sub_vars is not None
                            else set()
                        )

                        for missing_literal in missing_literals:
                            missing_pos_weight, missing_neg_weight = weights[
                                missing_literal
                            ]
                            missing_combined_weight = semiring.plus(
                                missing_pos_weight, missing_neg_weight
                            )
                            result_weight = semiring.times(
                                result_weight, missing_combined_weight
                            )

                    return result_weight

                elif node.is_false():
                    return semiring.zero()

                elif node.is_literal():
                    p_weight, n_weight = weights.get(abs(node.literal))
                    result_weight = p_weight if node.literal >= 0 else n_weight

                    # If smoothing, go over literals missed in scope
                    if smooth_flag:
                        lit_scope = {abs(node.literal)}

                        if expected_prime_vars is not None:
                            missing_literals = expected_prime_vars.difference(
                                lit_scope)
                        else:
                            missing_literals = set()
                        if expected_sub_vars is not None:
                            missing_literals |= expected_sub_vars.difference(
                                lit_scope)

                        for missing_literal in missing_literals:
                            missing_pos_weight, missing_neg_weight = weights[
                                missing_literal
                            ]
                            missing_combined_weight = semiring.plus(
                                missing_pos_weight, missing_neg_weight
                            )
                            result_weight = semiring.times(
                                result_weight, missing_combined_weight
                            )

                    return result_weight

                else:
                    raise Exception("Unknown leaf type for node {}".format(node))
            else:
                # Decision node
                if node is not None and not node.is_decision():
                    raise Exception(
                        "Expected a decision node for node {}".format(node))

                result_weight = None
                for prime_weight, sub_weight, prime_vars, sub_vars in rvalues:
                    branch_weight = semiring.times(prime_weight, sub_weight)

                    # If smoothing, go over literals missed in scope
                    if smooth_flag:
                        missing_literals = expected_prime_vars.difference(
                            prime_vars
                        ) | expected_sub_vars.difference(sub_vars)
                        for missing_literal in missing_literals:
                            missing_pos_weight, missing_neg_weight = weights[
                                missing_literal
                            ]
                            missing_combined_weight = semiring.plus(
                                missing_pos_weight, missing_neg_weight
                            )
                            branch_weight = semiring.times(
                                branch_weight, missing_combined_weight
                            )

                    # Add to current intermediate result
                    if result_weight is not None:
                        result_weight = semiring.plus(result_weight, branch_weight)
                    else:
                        result_weight = branch_weight
                return result_weight

        return func_weightedmodelcounting

    def get_wmc(self,
                sdd_manager,
                node,
                weights,
                semiring,
                wmc_func,
                literal=None,
                smooth_to_root=False,
                ):
        """Perform Weighted Model Count on the given node or the given literal.

        Common usage: wmc(node, weights, semiring) and wmc(node, weights, semiring, smooth_to_root=True)

        :param node: node to evaluate Type: SddNode
        :param weights: weights for the variables in the node. Type: {literal_id : (pos_weight, neg_weight)}
        :param semiring: use the operations defined by this semiring. Type: Semiring
        :param literal: When a literal is given, the result of WMC(literal) is returned instead.
        :param pr_semiring: Whether the given semiring is a (logspace) probability semiring.
        :param perform_smoothing: Whether to perform smoothing. When pr_semiring is True, smoothing is performed
            regardless.
        :param smooth_to_root: Whether to perform smoothing compared to the root. When pr_semiring is True, smoothing
            compared to the root is not performed regardless of this flag.
        :param wmc_func: The WMC function to use. If None, a built_in one will be used that depends on the given
            semiring. Type: function[SddNode, List[Tuple[prime_weight, sub_weight, Set[prime_used_lit],
            Set[sub_used_lit]]], Set[expected_prime_lit], Set[expected_sub_lit]] -> weight
        :type weights: dict[int, tuple[Any, Any]]
        :type semiring: Semiring
        :type pr_semiring: bool
        :type perform_smoothing: bool
        :type smooth_to_root: bool
        :type wmc_func: function
        :return: weighted model count of node if literal=None, else the weights are propagated up to node but the
            weighted model count of literal is returned.
        """
        try:
            varcount = sdd_manager.var_count()
            # Cover edge case e.g. node=SddNode(True)
            modified_weights = False
            if varcount == 1 and weights.get(1) is None:
                modified_weights = True
                weights[1] = (
                    semiring.one(),
                    semiring.zero(),
                )  # because 1 + 0 = 1 and 1 * x = x

            # Calculate result
            query_node = (
                node if literal is None else sdd_manager.literal(literal)
            )

            sdd_iterator = SddIterator(
                sdd_manager, smooth_to_root=smooth_to_root
            )
            result = sdd_iterator.depth_first(query_node, wmc_func)
            if weights.get(0) is not None:  # Times the weight of True
                result = semiring.times(result, weights[0][0])

            # Restore edge case modification
            if modified_weights:
                weights.pop(1)

            sdd_manager.set_prevent_transformation(prevent=False)
            return result

        except Exception as e:
            print("excetption")
            raise e
