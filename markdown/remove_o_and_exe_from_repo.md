  About your co-author's original repo: A PR from your fork to his repo will not rewrite his history — it would only add your new commits on top. To clean up the original repo, he'd need to run the same git
  filter-repo commands himself and force-push to main. You could send him the two commands:

  pip3 install git-filter-repo

  git filter-repo --path bin/ --path obj/ --invert-paths --force
  git filter-repo --path OneDrive/ --path-glob "*.o" --path-glob "*.a" --path-glob "*.d" --invert-paths --force

  git reflog expire --expire=now --all
  git gc --aggressive --prune=now
  git push origin main --force


