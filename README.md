Projeto de disciplina MACHINE LEARNING - Algoritimos Supervisionados.

Etapas:

1. Faça o módulo do [Kaggle Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning):
   Comprove a finalização do módulo com um print que contenha data e identificação do aluno.

   Trabalho com base:

   Iremos usar a base de dados de vinhos verdes portugueses (nas variantes branco e tinto) que encontra-se disponível no [Kaggle](https://www.kaggle.com/datasets/rajyellow46/wine-quality):

   **Informações do conjunto de dados:**

   O conjunto de dados foi baixado do UCI Machine Learning Repository.

   Os dois conjuntos de dados estão relacionados com as variantes tinto e branco do vinho português "Vinho Verde". A referência [Cortez et al., 2009]. Devido a questões de privacidade e logística, apenas variáveis físico-químicas (insumos) e sensoriais (saída) estão disponíveis (por exemplo, não há dados sobre tipos de uva, marca de vinho, preço de venda do vinho, etc.).

   Esses conjuntos de dados podem ser vistos como tarefas de classificação ou regressão. As classes são ordenadas e não equilibradas (por exemplo, há muito mais vinhos normais do que excelentes ou ruins). Algoritmos de detecção de outliers podem ser usados para detectar os poucos vinhos excelentes ou ruins. Além disso, não temos certeza se todas as variáveis de entrada são relevantes. Portanto, pode ser interessante testar métodos de seleção de recursos.

   Dois conjuntos de dados foram combinados e alguns valores foram removidos aleatoriamente.

   **Informações do atributo:**

   Para mais informações, leia [Cortez et al., 2009].
   Variáveis de entrada (com base em testes físico-químicos):
   1 - acidez fixa
   2 - acidez volátil
   3 - ácido cítric
   4 - açúcar residual
   5 - cloretos
   6 - dióxido de enxofre livre
   7 - dióxido de enxofre total
   8 - densidade
   9 - pH
   10 - sulfatos
   11 - álcool
   Variável de saída (com base em dados sensoriais):
   12 - qualidade (pontuação entre 0 e 10)

   - `fixed acidity`: a maioria dos ácidos envolvidos com vinho (não evaporam prontamente)
   - `volatile acidity`: a quantidade de ácido acético no vinho, que em níveis muito altos pode levar a um gosto desagradável de vinagre
   - `citric acid`: encontrado em pequenas quantidades, o ácido cítrico pode adicionar "leveza" e sabor aos vinhos
   - `residual sugar`: a quantidade de açúcar restante após a fermentação é interrompida, é raro encontrar vinhos com menos de 1 grama / litro e vinhos com mais de 45 gramas / litro são considerados doces
   - `chlorides`: a quantidade de sal no vinho
   - `free sulfur dioxide`: a forma livre de SO2 existe em equilíbrio entre o SO2 molecular (como gás dissolvido) e o íon bissulfito; impede o crescimento microbiano e a oxidação do vinho
   - `total sulfur dioxide`: Quantidade de formas livres e encadernadas de S02; em baixas concentrações, o SO2 é quase indetectável no vinho, mas nas concentrações de SO2 acima de 50 ppm, o SO2 se torna evidente no nariz e no sabor do vinho.
   - `density`: a densidade do vinho é próxima a da água, dependendo do percentual de álcool e teor de açúcar
   - `pH`: descreve se o vinho é ácido ou básico numa escala de 0 (muito ácido) a 14 (muito básico); a maioria dos vinhos está entre 3-4 na escala de pH
   - `sulphates`: um aditivo de vinho que pode contribuir para os níveis de gás de dióxido de enxofre (S02), que age como um antimicrobiano e antioxidante
   - `alcohol`: o percentual de álcool no vinho

   Existe ainda uma variável chamada `quality`. Essa variável é uma nota de qualidade do vinho que varia de 0 a 10.

   **
   Para as questões 2-5 usaremos apenas os vinhos do tipo "branco".

   **
2. Faça o download da base - esta é uma base real, apresentada no artigo:
   P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

   Ela possui uma variável denominada "quality", uma nota de 0 a 10 que denota a qualidade do vinho. Crie uma nova variável, chamada "opinion" que será uma variável categórica igual à 0, quando quality for menor e igual à 5. O valor será 1, caso contrário. Desconsidere a variável quality para o restante da análise.
3. Descreva as variáveis presentes na base. Quais são as variáveis? Quais são os tipos de variáveis (discreta, categórica, contínua)? Quais são as médias e desvios padrões?
4. Com a base escolhida:

   1. Descreva as etapas necessárias para criar um modelo de classificação eficiente.
   2. Treine um modelo de regressão logística usando um modelo de validação cruzada estratificada com k-folds (k=10) para realizar a classificação. Calcule para a base de teste:
      i. a média e desvio da acurácia dos modelos obtidos;
      ii. a média e desvio da precisão dos modelos obtidos;
      iii. a média e desvio da recall dos modelos obtidos;
      iv. a média e desvio do f1-score dos modelos obtidos.
   3. Treine um modelo de árvores de decisão usando um modelo de validação cruzada estratificada com k-folds (k=10) para realizar a classificação. Calcule para a base de teste:
      i. a média e desvio da acurácia dos modelos obtidos;
      ii. a média e desvio da precisão dos modelos obtidos;
      iii. a média e desvio da recall dos modelos obtidos;
      iv. a média e desvio do f1-score dos modelos obtidos.
   4. Treine um modelo de SVM usando um modelo de validação cruzada estratificada com k-folds (k=10) para realizar a classificação. Calcule para a base de teste:
      i. a média e desvio da acurácia dos modelos obtidos;
      ii. a média e desvio da precisão dos modelos obtidos;
      iii. a média e desvio da recall dos modelos obtidos;
      iv. a média e desvio do f1-score dos modelos obtidos.
5. Em relação à questão anterior, qual o modelo deveria ser escolhido para uma eventual operação. Responda essa questão mostrando a comparação de todos os modelos, usando um gráfico mostrando a curva ROC média para cada um dos gráficos e justifique.
6. Com a escolha do melhor modelo, use os dados de vinho tinto, presentes na base original e faça a inferência (não é para treinar novamente!!!) para saber quantos vinhos são bons ou ruins. Utilize o mesmo critério utilizado com os vinhos brancos, para comparar o desempenho do modelo. Ele funciona da mesma forma para essa nova base? Justifique.
7. Disponibilize os códigos usados para responder da questão 2-6 em uma conta github e indique o link para o repositório.
