# Upload Results to Evaluation Server

This page describes the upload instructions for submitting results to the
evaluation servers for the PoseTrack DensePose challenge. Submitting results allows
you to participate in the challenges and compare results to the
state-of-the-art on the public leaderboards. Note that you can obtain results
on val by running the
[evaluation code](https://github.com/facebookresearch/DensePose/blob/master/detectron/datasets/densepose_cocoeval.py)
locally. One can also take advantage of the
[vkhalidov/densepose-codalab](https://hub.docker.com/r/vkhalidov/densepose-codalab/)
docker image which was tailored specifically for evaluation.
Submitting to the evaluation server provides results on the val and
test sets. We now give detailed instructions for submitting to the evaluation
server:

1. Create an account on CodaLab. This will allow you to participate in
PoseTrack DensePose challenge.

2. Carefully review the [guidelines](readme.md) for
entering the PoseTrack DensePose challenge and using the test sets.
We emphasize that any form of **annotation or use of the test sets
for supervised or unsupervised training is strictly forbidden**.

3. Prepare a JSON file containing your results in the correct
[results format](results_format.md).

4. File naming: the JSON file should be named `posetrack_[subset]_[alg]_results.json`.
Replace `[subset]` with the subset you are using (`val` or `test`),
and `[alg]` with your algorithm name. Finally, place the JSON
file into a zip file named `posetrack_[subset]_[alg]_results.zip`.

5. To submit your zipped result file to the PoseTrack DensePose Challenge, click on
the “Participate” tab on the
[CodaLab evaluation server](https://competitions.codalab.org/competitions/19650) page.
When you select “Submit / View Results” on the left panel, you will be able to choose
the subset. Please fill in the required fields and click “Submit”. A pop-up will
prompt you to select the results zip file for upload. After the file is uploaded
the evaluation server will begin processing. To view the status of your submission
please select “Refresh Status”. Please be patient, the evaluation may take quite
some time to complete (from ~20m to a few hours). If the status of your submission
is “Failed” please check your file is named correctly and has the right format.

6. Please enter submission information into Codalab. The most important fields
are "Team name", "Method description", and "Publication URL", which are used
to populate the leaderboard. Additionally, under "user setting" in the
upper right, please add "Team members". There have been issues with the
"Method Description", we may collect these via email if necessary. These
settings are not intuitive, but we have no control of the Codalab website.
For the "Method description" we encourage participants to give detailed
method information that will help
the award committee invite participants with the most innovative methods.
**Listing external data used is mandatory.** You may also consider giving some
basic performance breakdowns on test subset (e.g., single model versus
ensemble results), runtime, or any other information you think may be pertinent
to highlight the novelty or efficacy of your method.

7. After you submit your results to the test eval server, you can control
whether your results are publicly posted to the CodaLab leaderboard. To toggle
the public visibility of your results please select either “post to leaderboard”
or “remove from leaderboard”. Only one result can be published to the leaderboard
at any time.

8. After evaluation is complete and the server shows a status of “Finished”,
you will have the option to download your evaluation results by selecting
“Download evaluation output from scoring step.” The zip file will contain the
score file `scores.txt`.

