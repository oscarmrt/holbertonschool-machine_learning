-- Write a SQL script that creates a stored procedure ComputeAverageScoreForUser
-- that computes and store the average score for a student.
-- Requirements:
-- Procedure ComputeAverageScoreForUser is taking 1 input:
-- user_id, a users.id value (you can assume user_id is linked to an existing users)

delimiter //
CREATE PROCEDURE ComputeAverageScoreForUser(
	IN user_id_new INTEGER)
	BEGIN
		UPDATE users SET average_score=(
			SELECT AVG(score) FROM corrections WHERE user_id=user_id_new)
			WHERE id=user_id_new;
	END //
delimiter ;
