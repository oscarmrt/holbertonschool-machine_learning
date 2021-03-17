-- Write a SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans
-- Requirements:
-- Import this table dump: metal_bands.sql.zip
-- Column names must be: origin and nb_fans
-- Your script can be executed on any database
-- Context: Calculate/compute something is always power intensive… better to distribute the load!

SELECT origin, sum(fans) AS nb_fans FROM metal_bands GROUP BY origin ORDER BY nb_fans DESC;
