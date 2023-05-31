from pysdd.sdd import SddManager, Vtree
import math
import sys
import os

class WMC():

    def __init__(self,  k=5, delta=0.01):
        # topk
        self.k = 5
        self.delta = delta
        self.sdd = None

   
    
    def solve(self, answers, probabilities):
        '''Simple off the shelve WMC solver, this method has/stores no WMC semiring '''
        ### Now used for quick inference, but botch WMC methods should be merged. (see WMCCalculator) ###
        wmc_per_answer = {}
        answers, variables_per_answer = answers

        # for _, lineage in answers.items():
        #     print("proof", lineage)
        try:
            for answer, lineage in answers.items():
                # print("answer", answer)
                # print("LEN LINEAGE", lineage)
                if len(lineage) > 1:
                    number_of_variables = len(variables_per_answer[answer])
                    vtree = Vtree(var_count=number_of_variables, var_order=list(
                        range(1, number_of_variables + 1)), vtree_type="balanced")
                    sddmanager = SddManager.from_vtree(vtree)

                    tupleToLiteral = {top_objects: sddmanager.literal(
                        index+1) for index, top_objects in enumerate(variables_per_answer[answer])}
                    literalToTuple = {v.literal: k for k,
                                      v in tupleToLiteral.items()}

                    for index, top_objects in enumerate(variables_per_answer[answer]):
                        literal = sddmanager.literal(index + 1)
                        tupleToLiteral[top_objects] = literal
                        literalToTuple[literal.literal] = top_objects

                    formula = None
                    for i, t_disjunct in enumerate(lineage):
                        conjunction = None
                        for conjunct in t_disjunct:

                            if len(conjunct) < 1:
                                continue
                            try:
                                literal = tupleToLiteral[tuple(conjunct)]
                            
                            except Exception as e:
                                print("exception", e)

                            if conjunction is None:
                                conjunction = literal
                            else:
                                conjunction = conjunction.conjoin(literal)
                        if formula is None:
                            formula = conjunction
                        else:
                            formula = formula.disjoin(conjunction)
                    wmc = formula.wmc(log_mode=True)

                    for literal in tupleToLiteral.values():
                        # Positive literal weight
                        fact = literalToTuple[literal.literal]
                        if not fact:
                            continue
                        try:
                            weight = probabilities[fact]
                        except KeyError:
                            
                            weight = 1.0

                        try:
                            if weight == 1.0:
                                wmc.set_literal_weight(literal, wmc.one_weight)
                                wmc.set_literal_weight(-literal, wmc.zero_weight)


                            elif weight == 0.0:
                                wmc.set_literal_weight(literal, wmc.zero_weight)
                                wmc.set_literal_weight(-literal, wmc.zero_weight)


                            else:
                                wmc.set_literal_weight(literal, math.log(weight))
                                # Negative literal weight
                                wmc.set_literal_weight(-literal, math.log((1-weight)))
                    
                        except Exception as e:
                            raise e
                            

                    w = wmc.propagate()
                    probability = math.exp(w)

                else:
                    probability = 1.0
                    t_disjunct = lineage[0]
                    # lookup probs of fact pairs
                    fact = None
                    for fact in t_disjunct:
                        if not fact:
                            continue
                        try:
                            weight = probabilities[fact]
                        except KeyError:
                            # print("KEY ERROR EXCEPTION")
                            weight = 1.0

                        except Exception as e:
                            print("THE EX", e)
                            raise e

                        probability = probability * weight

                wmc_per_answer[answer] = probability

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            raise e

        
        
        return wmc_per_answer

  