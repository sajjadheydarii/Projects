
DECLARE
	@StartDate DATE = '2020-01-01',
	@EndDate DATE = '2020-02-02'

;WITH CTE AS
(
	SELECT *
	FROM  
	(
		SELECT 
			Q.UserID, 
			Q.OrderId, 
			ROW_NUMBER() OVER (PARTITION BY Q.UserID ORDER BY Q.OrderDate) RN
		FROM  Q
		WHERE
			Q.Payment = 1 
			AND EXISTS (SELECT 1 
				FROM  Q1 
				WHERE 
					Q1.UserType = 'New' 
					AND CAST(Q1.OrderDate AS DATE) BETWEEN @StartDate AND @EndDate
					AND Q1.UserID = Q.UserID)
	) AS SourceTable  
	PIVOT  
	(  
		MIN(OrderId)  
		FOR RN IN ([1], [2], [3])  
	) AS PivotTable
)
SELECT 
	CTE.UserID, 
	DATEDIFF(day,Q1.QuestionPaymentDate,Q2.QuestionPaymentDate) AS Order1_2_GapDay,
	DATEDIFF(day,Q1.QuestionPaymentDate,Q3.QuestionPaymentDate) AS Order1_3_GapDay,

FROM
	CTE INNER JOIN 
	CON.FactQuestion Q1 ON Q1.QuestionId = CTE.[1] LEFT OUTER JOIN
	CON.FactQuestion Q2 ON Q2.QuestionId = CTE.[2] LEFT OUTER JOIN
	CON.FactQuestion Q3 ON Q3.QuestionId = CTE.[3];
