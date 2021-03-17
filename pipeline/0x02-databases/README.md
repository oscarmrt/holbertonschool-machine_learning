This project is about 0x02. Databases
0-create_database_if_missing.sql - Write a script that creates the database db_0 in your MySQL server.
If the database db_0 already exists, your script should not fail
You are not allowed to use the SELECT or SHOW statements

1-first_table.sql - Write a script that creates a table called first_table in the current database in your MySQL server.
first_table description:
id INT
name VARCHAR(256)
The database name will be passed as an argument of the mysql command
If the table first_table already exists, your script should not fail
You are not allowed to use the SELECT or SHOW statements

2-list_values.sql - Write a script that lists all rows of the table first_table in your MySQL server.
All fields should be printed
The database name will be passed as an argument of the mysql command

3-insert_value.sql - Write a script that inserts a new row in the table first_table in your MySQL server.
New row:
id = 89
name = Holberton School
The database name will be passed as an argument of the mysql command

4-best_score.sql - Write a script that lists all records with a score >= 10 in the table second_table in your MySQL server.
Results should display both the score and the name (in this order)
Records should be ordered by score (top first)
The database name will be passed as an argument of the mysql command

5-average.sql - Write a script that computes the score average of all records in the table second_table in your MySQL server.
The result column name should be average
The database name will be passed as an argument of the mysql command

6-avg_temperatures.sql - Import in hbtn_0c_0 database this table dump: download
Write a script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).

7-max_state.sql - Import in hbtn_0c_0 database this table dump: download (same as Temperatures #0)
Write a script that displays the max temperature of each state (ordered by State name).

8-genre_id_by_show.sql - Import the database dump from hbtn_0d_tvshows to your MySQL server: download
Write a script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked.
Each record should display: tv_shows.title - tv_show_genres.genre_id
Results must be sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
You can use only one SELECT statement
The database name will be passed as an argument of the mysql command

9-no_genre.sql - Import the database dump from hbtn_0d_tvshows to your MySQL server: download
Write a script that lists all shows contained in hbtn_0d_tvshows without a genre linked.
Each record should display: tv_shows.title - tv_show_genres.genre_id
Results must be sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
You can use only one SELECT statement
The database name will be passed as an argument of the mysql command

10-count_shows_by_genre.sql - Import the database dump from hbtn_0d_tvshows to your MySQL server: download
Write a script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
Each record should display: <TV Show genre> - <Number of shows linked to this genre>
First column must be called genre
Second column must be called number_of_shows
Don’t display a genre that doesn’t have any shows linked
Results must be sorted in descending order by the number of shows linked
You can use only one SELECT statement
The database name will be passed as an argument of the mysql command

11-rating_shows.sql - Import the database hbtn_0d_tvshows_rate dump to your MySQL server: download
Write a script that lists all shows from hbtn_0d_tvshows_rate by their rating.
Each record should display: tv_shows.title - rating sum
Results must be sorted in descending order by the rating
You can use only one SELECT statement
The database name will be passed as an argument of the mysql command

12-rating_genres.sql - Import the database dump from hbtn_0d_tvshows_rate to your MySQL server: download
Write a script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.
Each record should display: tv_genres.name - rating sum
Results must be sorted in descending order by their rating
You can use only one SELECT statement
The database name will be passed as an argument of the mysql command

13-uniq_users.sql - Write a SQL script that creates a table users following these requirements:
With these attributes:
id, integer, never null, auto increment and primary key
email, string (255 characters), never null and unique
name, string (255 characters)
If the table already exists, your script should not fail
Your script can be executed on any database
Context: Make an attribute unique directly in the table schema will enforced your business rules and avoid bugs in your application

14-country_users.sql - Write a SQL script that creates a table users following these requirements:
With these attributes:
id, integer, never null, auto increment and primary key
email, string (255 characters), never null and unique
name, string (255 characters)
country, enumeration of countries: US, CO and TN, never null (= default will be the first element of the enumeration, here US)
If the table already exists, your script should not fail
Your script can be executed on any database

15-fans.sql - Write a SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans
Requirements:
Import this table dump: metal_bands.sql.zip
Column names must be: origin and nb_fans
Your script can be executed on any database
Context: Calculate/compute something is always power intensive… better to distribute the load!

16-glam_rock.sql - Write a SQL script that lists all bands with Glam rock as their main style, ranked by their longevity
Requirements:
Import this table dump: metal_bands.sql.zip
Column names must be: band_name and lifespan (in years)
You should use attributes formed and split for computing the lifespan
Your script can be executed on any database

17-store.sql - Write a SQL script that creates a trigger that decreases the quantity of an item after adding a new order.
Quantity in the table items can be negative.
Context: Updating multiple tables for one action from your application can generate issue: network disconnection, crash, etc… to keep your data in a good shape, let MySQL do it for you!

18-valid_email.sql - Write a SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.
Context: Nothing related to MySQL, but perfect for user email validation - distribute the logic to the database itself!

19-bonus.sql - Write a SQL script that creates a stored procedure AddBonus that adds a new correction for a student.
Requirements:
Procedure AddBonus is taking 3 inputs (in this order):
user_id, a users.id value (you can assume user_id is linked to an existing users)
project_name, a new or already exists projects - if no projects.name found in the table, you should create it
score, the score value for the correction
Context: Write code in SQL is a nice level up!

20-average_score.sql - Write a SQL script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
Requirements:
Procedure ComputeAverageScoreForUser is taking 1 input:
user_id, a users.id value (you can assume user_id is linked to an existing users)

21-div.sql - Write a SQL script that creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.
Requirements:
You must create a function
The function SafeDiv takes 2 arguments:
a, INT
b, INT
And returns a / b or 0 if b == 0

22-list_databases - Write a script that lists all databases in MongoDB.

23-use_or_create_database - Write a script that creates or uses the database my_db

24-insert - Write a script that inserts a document in the collection school:
The document must have one attribute name with value “Holberton school”
The database name will be passed as option of mongo command

25-all - Write a script that lists all documents in the collection school:
The database name will be passed as option of mongo command

26-match - Write a script that lists all documents with name="Holberton school" in the collection school:
The database name will be passed as option of mongo command

27-count - Write a script that displays the number of documents in the collection school:
The database name will be passed as option of mongo command

28-update - Write a script that adds a new attribute to a document in the collection school:
The script should update only document with name="Holberton school" (all of them)
The update should add the attribute address with the value “972 Mission street”
The database name will be passed as option of mongo command

29-delete - Write a script that deletes all documents with name="Holberton school" in the collection school:
The database name will be passed as option of mongo command

30-all.py - Write a Python function that lists all documents in a collection:
Prototype: def list_all(mongo_collection):
Return an empty list if no document in the collection
mongo_collection will be the pymongo collection object

31-insert_school.py - Write a Python function that inserts a new document in a collection based on kwargs:
Prototype: def insert_school(mongo_collection, **kwargs):
mongo_collection will be the pymongo collection object
Returns the new _id

32-update_topics.py - Write a Python function that changes all topics of a school document based on the name:
Prototype: def update_topics(mongo_collection, name, topics):
mongo_collection will be the pymongo collection object
name (string) will be the school name to update
topics (list of strings) will be the list of topics approached in the school

33-schools_by_topic.py - Write a Python function that returns the list of school having a specific topic:
Prototype: def schools_by_topic(mongo_collection, topic):
mongo_collection will be the pymongo collection object
topic (string) will be topic searched

34-log_stats.py - Write a Python script that provides some stats about Nginx logs stored in MongoDB:
Database: logs
Collection: nginx
Display (same as the example):
first line: x logs where x is the number of documents in this collection
second line: Methods:
5 lines with the number of documents with the method = ["GET", "POST", "PUT", "PATCH", "DELETE"] in this order (see example below - warning: it’s a tabulation before each line)
one line with the number of documents with:
method=GET
path=/status
You can use this dump as data sample: dump.zip
The output of your script must be exactly the same as the example
