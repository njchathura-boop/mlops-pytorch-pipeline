# Screenshot Checklist

Capture these screenshots as you complete the work:

1. GitHub repository structure
2. `main`, `develop`, and feature branches
3. At least four merged Pull Requests
4. Successful GitHub Actions test
5. Local training JSON logs
6. Generated `classifier_v1.pt` checkpoint
7. Local `/health` response
8. Local `/predict` response
9. Training Docker image build
10. Training container output
11. Serving Docker image build
12. Serving container `/health` and `/predict`
13. `kubectl get jobs -n ml-training`
14. Completed training Job logs
15. `kubectl get pods -n ml-training`
16. `kubectl describe deployment model-serving -n ml-training`
17. `kubectl get svc -n ml-training`
18. `kubectl get hpa -n ml-training`
19. Successful Kubernetes port-forward prediction
