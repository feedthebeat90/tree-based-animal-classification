# Project Content and Style Guide Checklist

This document is a checklist you should use when 1) creating your own content and 2) giving feedback on your fellow intern's project.

**When giving feedback on a project, you should also give feedback on technical correctness of code, technical correctness of statistical tecniques, code aesthetics, flow of writing, storytelling improvements, etc. that are beyond the scope of this content and style guide checklist.**

_Note: you can check for some of the following character limit warnings in the build log in Teach, though some of them are out of date for Projects V2. Where there is discrepancy, use this rubric as the source of truth._

## General

- [ ] Is the [style guide](https://instructor-support.datacamp.com/en/articles/2584383-style-guidelines) adhered to?
  - [ ] Use American English.
  - [ ] Parentheses after function/method names.
  - [ ] Format package names as inline code.
  - [ ] Programming language (i.e., Python, R, or SQL) style guide (links in style guide link above) adhered to.
  - [ ] Use en-dashes (–) for numeric ranges (e.g., 18–64).
  - Code comments:
     - [ ] Code comments start on a new line, not on the same line as the code.
     - [ ] A single space separates the comment character (e.g. `#`) and the first word of the comment.
     - [ ] The first letter of every comment is capitalized.
     - [ ] Comments are <= 75 characters.
     - [ ] If your comment is one sentence, it doesn't have ending punctuation.
     - [ ] If you have multiple sentences in your comment, it ends with a period.
     - [ ] Backticks or quotes aren't used to refer to variables or functions inside comments.
     - [ ] Code comments are identical for solution code and the hint.
    - [ ] All instructions have ending punctuation (even if the last word is formatted inline code).
    - [ ] The "Helpful links" bullets should not be full sentences and should not have ending punctuation.

## type:NotebookTask (i.e., task titles)
- [ ] Task titles are <= 55 characters.
- [ ] Task titles are written in sentence case.

## Context (`@context`)

- [ ] Are there <= 1200 characters and <= 4 paragraphs in the context cell for Task 1? (You _can't_ check for this in the Teach build warnings.)
- [ ] Are there <= 800 characters and <= 3 paragraphs in the context cells for Tasks 2-end? (You _can_ check for this in the Teach build warnings.)
- [ ] Is there an image embedded in the context cell for Task 1?
- [ ] The context cells emphasize the real-world impact of this data and tell the story within the data, instead of simply overviewing the data science techniques used.
- [ ] There are **no** references to "you", "project", or the student in the context cells. Bad: _"In this project, you will explore..."_ Good: _"In this notebook, we will explore..."_

## Solution Code (`@solution`)
- [ ] There are <= 15 lines of code in each solution code cell (including comments).
- [ ] All lines of code are <= 75 characters.
- [ ] A code comment exists for each instruction bullet, at minimum.
- [ ] Code comments [explain the why](https://blog.codinghorror.com/code-tells-you-how-comments-tell-you-why/) of the code.

## Instructions (`@instructions`)
- [ ] The instructions cell for Task 1 is <= 1200 characters.
- [ ] The instructions cells for Tasks 2-end are <= 700 characters.
- [ ] Each instruction relates to one "chunk" of code, where a "chunk" can be >= 1 line of code and corresponds to a complete task you ask the student to do. E.g., _"Create a plot with [this column] on the x-axis and [that column] on the y-axis."_
- [ ] There are <= 4 instructions bullets.
- [ ] Each instructions bullet has <= 2 sentences.
- [ ] There are **no** references to "we" in the instructions cells. Bad: _"In this task, we will load the data."_ Good: _"Load the data."_ Note: you can use "you" if you need to.
- [ ] The helpful links section has <= 1 DataCamp exercise linked in it.
- [ ] The helpful links section has <= 1 function/method documentation page linked in it.
- [ ] If the solution code for a task produces output, a screenshot of the expected output is included at the bottom of that task's instructions.
- [ ] If the solution code for a task doesn't produce output, the bottom of the instructions for that task says "**There is no expected output for this task.**"

## Sample Code (`@sample_code`)

- [ ] Each task's sample code cell has a comment that says `# ... YOUR CODE FOR TASK [x] ...` and nothing else.

## Hint (`@hint`)

- David will complete this section of the rubric later. :)

## Tests (`@tests`)

- David will complete this section of the rubric later. :)