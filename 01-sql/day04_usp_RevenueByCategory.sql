CREATE   PROCEDURE dbo.usp_RevenueByCategory @Year INT, @TopN INT = 5 AS BEGIN SET NOCOUNT ON; SELECT TOP (@TopN) Category, SUM(SalesAmount) AS Revenue, SUM(Profit) AS Profit FROM dbo.vw_SalesDetail WHERE Yr = @Year GROUP BY Category ORDER BY Revenue DESC; END

(1 row affected)
