import plotly.figure_factory as ff

df = [dict(Task="Implemening a classifier on COMPASS", Start='2020-02-12', Finish='2020-02-19'),
      dict(Task="Implementing a different classifier and having a candidate for a final classifier for the project", Start='2020-02-19', Finish='2020-02-26'),
      dict(Task="Using correction algorihm for biases and checking that it works", Start='2020-02-26', Finish='2020-03-10'),
	  dict(Task="Midway evaluation (Report writing - Introduction, data and methods)", Start='2020-02-12', Finish='2020-03-18'),
	  dict(Task="Giving and recieving written feedback", Start='2020-03-18', Finish='2020-03-25'),
	  dict(Task="Finnishing Results and coding", Start='2020-03-25', Finish='2020-04-15'),
	  dict(Task="Report Writing", Start='2020-04-15', Finish='2020-06-07'),
	  dict(Task="Group Meetings", Start='2020-02-12', Finish='2020-06-08'),
	  dict(Task="Three week period - finnishing report as well as preparing an oral pitch", Start='2020-06-08', Finish='2020-06-25'),
	  dict(Task='"Faglig Fest"', Start='2020-06-25', Finish='2020-06-26'),
	  ]

fig = ff.create_gantt(df)
fig.show()