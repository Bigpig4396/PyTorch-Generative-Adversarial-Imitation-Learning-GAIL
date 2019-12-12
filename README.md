# PyTorch-Generative-Adversarial-Imitation-Learning-GAIL

This is a pytorch implementation of GAIL in a navigation problem. Just run "GAIL_OppositeV4.py". The program first train a expert to solve the task, then sample a expert trajectory, then use GAIL to imitate it.


You will get something like that at the end
 ```
expert trajectory
step 0 agent 1 at [[0.44444444 0.11111111]] agent 1 action 0
step 1 agent 1 at [[0.33333333 0.11111111]] agent 1 action 0
step 2 agent 1 at [[0.22222222 0.11111111]] agent 1 action 0
step 3 agent 1 at [[0.11111111 0.11111111]] agent 1 action 3
step 4 agent 1 at [[0.11111111 0.22222222]] agent 1 action 3
step 5 agent 1 at [[0.11111111 0.33333333]] agent 1 action 3
step 6 agent 1 at [[0.11111111 0.44444444]] agent 1 action 3
step 7 agent 1 at [[0.11111111 0.55555556]] agent 1 action 3
step 8 agent 1 at [[0.11111111 0.66666667]] agent 1 action 3
step 9 agent 1 at [[0.11111111 0.77777778]] agent 1 action 1
step 10 agent 1 at [[0.22222222 0.77777778]] agent 1 action 1
step 11 agent 1 at [[0.33333333 0.77777778]] agent 1 action 1
learnt trajectory
step 0 agent 1 at [[0.44444444 0.11111111]] agent 1 action 0
step 1 agent 1 at [[0.33333333 0.11111111]] agent 1 action 0
step 2 agent 1 at [[0.22222222 0.11111111]] agent 1 action 0
step 3 agent 1 at [[0.11111111 0.11111111]] agent 1 action 3
step 4 agent 1 at [[0.11111111 0.22222222]] agent 1 action 3
step 5 agent 1 at [[0.11111111 0.33333333]] agent 1 action 3
step 6 agent 1 at [[0.11111111 0.44444444]] agent 1 action 3
step 7 agent 1 at [[0.11111111 0.55555556]] agent 1 action 3
step 8 agent 1 at [[0.11111111 0.66666667]] agent 1 action 3
step 9 agent 1 at [[0.11111111 0.77777778]] agent 1 action 1
step 10 agent 1 at [[0.22222222 0.77777778]] agent 1 action 1
step 11 agent 1 at [[0.33333333 0.77777778]] agent 1 action 1
```
You can see it replicate the expert trajectory correctly.
