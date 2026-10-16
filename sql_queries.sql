-- sites
SELECT LaunchSite, COUNT(*) launches, SUM(Class) successful, ROUND(AVG(Class)*100,1) success_pct, ROUND(SUM(PayloadMass),1) payload_kg FROM launches GROUP BY LaunchSite ORDER BY launches DESC;

-- years
SELECT SUBSTR(Date,1,4) year, COUNT(*) launches, SUM(Class) successful, ROUND(AVG(Class)*100,1) success_pct FROM launches GROUP BY year ORDER BY year;

-- orbits
SELECT Orbit, COUNT(*) launches, SUM(Class) successful, ROUND(AVG(Class)*100,1) success_pct FROM launches GROUP BY Orbit ORDER BY launches DESC;

