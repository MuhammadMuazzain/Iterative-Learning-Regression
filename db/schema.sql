-- Database schema for AI Text Detection System
-- Run this script to set up the database

-- Create database (run this as superuser)
-- CREATE DATABASE ai_text_feedback;

-- Connect to the database
\c ai_text_feedback;

-- Create feedback_events table
CREATE TABLE IF NOT EXISTS feedback_events (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    text_hash VARCHAR(64) UNIQUE NOT NULL,
    model_score FLOAT NOT NULL,
    user_feedback VARCHAR(20) NOT NULL CHECK (user_feedback IN ('human', 'ai', 'human_edited_ai')),
    model_version VARCHAR(100),
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_score FLOAT  -- Computed: 0.0 for human, 0.5 for human_edited_ai, 1.0 for ai
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_text_hash ON feedback_events(text_hash);
CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback_events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_user_feedback ON feedback_events(user_feedback);

-- Function to automatically set target_score based on user_feedback
CREATE OR REPLACE FUNCTION set_target_score()
RETURNS TRIGGER AS $$
BEGIN
    NEW.target_score := CASE 
        WHEN NEW.user_feedback = 'human' THEN 0.0
        WHEN NEW.user_feedback = 'human_edited_ai' THEN 0.5
        WHEN NEW.user_feedback = 'ai' THEN 1.0
        ELSE NULL
    END;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically populate target_score on insert
DROP TRIGGER IF EXISTS trigger_set_target_score ON feedback_events;
CREATE TRIGGER trigger_set_target_score
    BEFORE INSERT ON feedback_events
    FOR EACH ROW
    EXECUTE FUNCTION set_target_score();

-- Update existing records without target_score (if any)
UPDATE feedback_events 
SET target_score = CASE 
    WHEN user_feedback = 'human' THEN 0.0
    WHEN user_feedback = 'human_edited_ai' THEN 0.5
    WHEN user_feedback = 'ai' THEN 1.0
    ELSE NULL
END
WHERE target_score IS NULL;

-- Verify table structure
\d feedback_events

