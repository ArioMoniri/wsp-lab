Vendored Google IFEval instruction checkers from the installed lm-eval 0.4.13
(`lm_eval/tasks/ifeval`), originally Google Research's
instruction_following_eval. Apache-2.0 headers are retained in source files.

Local changes: imports target wsp.eval.ifeval; formatting follows repository
ruff settings. The strict wrapper evaluates every instruction and strips the
model's separate thinking block before checking its final response.

Upstream: https://github.com/google-research/google-research/tree/master/instruction_following_eval
Harness: https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13/lm_eval/tasks/ifeval
