input -> hidden layer -> output
    |       |          +
    ->  cerebellum     |

21 feature = yksi joint x DoF (3 tai 1). Output 21 = force input jokaseen niveleen.
Reward = liike eteenpäin. Vaikuttaa kaikkiin paitsi cerebellumiin.
Error  = kertoo kuinka hyvin edellinen toiminta vaikutti. Vaikuttaa ainoastaan cerebellumiin.

Add on script that downloads weights to a file. Add a script reads weights from a file.