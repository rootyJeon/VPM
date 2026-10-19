<p align="center">
  <img src="docs/vpm-icon.svg" alt="VPM" width="120">
</p>

<h1 align="center">Learning the 4D World through Point Trajectory Forecasting</h1>

<p align="center">
  <a href="https://rootyjeon.github.io/video-point-model/"><img src="https://img.shields.io/badge/Project-Page-1f6feb.svg?logo=googlechrome&logoColor=white" alt="Project page"></a>
  <a href="https://huggingface.co/datasets/rooty2020/VPM-Data"><img src="https://img.shields.io/badge/VPM--Data-Dataset-FFD21E.svg?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5NSIgaGVpZ2h0PSI4OCIgZmlsbD0ibm9uZSI%2BPHBhdGggZmlsbD0iI0ZGRDIxRSIgZD0iTTQ3LjIxIDc2LjVhMzQuNzUgMzQuNzUgMCAxIDAgMC02OS41IDM0Ljc1IDM0Ljc1IDAgMCAwIDAgNjkuNVoiLz48cGF0aCBmaWxsPSIjRkY5RDBCIiBkPSJNODEuOTYgNDEuNzVhMzQuNzUgMzQuNzUgMCAxIDAtNjkuNSAwIDM0Ljc1IDM0Ljc1IDAgMCAwIDY5LjUgMFptLTczLjUgMGEzOC43NSAzOC43NSAwIDEgMSA3Ny41IDAgMzguNzUgMzguNzUgMCAwIDEtNzcuNSAwWiIvPjxwYXRoIGZpbGw9IiMzQTNCNDUiIGQ9Ik01OC41IDMyLjNjMS4yOC40NCAxLjc4IDMuMDYgMy4wNyAyLjM4YTUgNSAwIDEgMC02Ljc2LTIuMDdjLjYxIDEuMTUgMi41NS0uNzIgMy43LS4zMlpNMzQuOTUgMzIuM2MtMS4yOC40NC0xLjc5IDMuMDYtMy4wNyAyLjM4YTUgNSAwIDEgMSA2Ljc2LTIuMDdjLS42MSAxLjE1LTIuNTYtLjcyLTMuNy0uMzJaIi8%2BPHBhdGggZmlsbD0iI0ZGMzIzRCIgZD0iTTQ2Ljk2IDU2LjI5YzkuODMgMCAxMy04Ljc2IDEzLTEzLjI2IDAtMi4zNC0xLjU3LTEuNi00LjA5LS4zNi0yLjMzIDEuMTUtNS40NiAyLjc0LTguOSAyLjc0LTcuMTkgMC0xMy02Ljg4LTEzLTIuMzhzMy4xNiAxMy4yNiAxMyAxMy4yNloiLz48cGF0aCBmaWxsPSIjM0EzQjQ1IiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0zOS40MyA1NGE4LjcgOC43IDAgMCAxIDUuMy00LjQ5Yy40LS4xMi44MS41NyAxLjI0IDEuMjguNC42OC44MiAxLjM3IDEuMjQgMS4zNy40NSAwIC45LS42OCAxLjMzLTEuMzUuNDUtLjcuODktMS4zOCAxLjMyLTEuMjVhOC42MSA4LjYxIDAgMCAxIDUgNC4xN2MzLjczLTIuOTQgNS4xLTcuNzQgNS4xLTEwLjcgMC0yLjM0LTEuNTctMS42LTQuMDktLjM2bC0uMTQuMDdjLTIuMzEgMS4xNS01LjM5IDIuNjctOC43NyAyLjY3cy02LjQ1LTEuNTItOC43Ny0yLjY3Yy0yLjYtMS4yOS00LjIzLTIuMS00LjIzLjI5IDAgMy4wNSAxLjQ2IDguMDYgNS40NyAxMC45N1oiIGNsaXAtcnVsZT0iZXZlbm9kZCIvPjxwYXRoIGZpbGw9IiNGRjlEMEIiIGQ9Ik03MC43MSAzN2EzLjI1IDMuMjUgMCAxIDAgMC02LjUgMy4yNSAzLjI1IDAgMCAwIDAgNi41Wk0yNC4yMSAzN2EzLjI1IDMuMjUgMCAxIDAgMC02LjUgMy4yNSAzLjI1IDAgMCAwIDAgNi41Wk0xNy41MiA0OGMtMS42MiAwLTMuMDYuNjYtNC4wNyAxLjg3YTUuOTcgNS45NyAwIDAgMC0xLjMzIDMuNzYgNy4xIDcuMSAwIDAgMC0xLjk0LS4zYy0xLjU1IDAtMi45NS41OS0zLjk0IDEuNjZhNS44IDUuOCAwIDAgMC0uOCA3IDUuMyA1LjMgMCAwIDAtMS43OSAyLjgyYy0uMjQuOS0uNDggMi44LjggNC43NGE1LjIyIDUuMjIgMCAwIDAtLjM3IDUuMDJjMS4wMiAyLjMyIDMuNTcgNC4xNCA4LjUyIDYuMSAzLjA3IDEuMjIgNS44OSAyIDUuOTEgMi4wMWE0NC4zMyA0NC4zMyAwIDAgMCAxMC45MyAxLjZjNS44NiAwIDEwLjA1LTEuOCAxMi40Ni01LjM0IDMuODgtNS42OSAzLjMzLTEwLjktMS43LTE1LjkyLTIuNzctMi43OC00LjYyLTYuODctNS03Ljc3LS43OC0yLjY2LTIuODQtNS42Mi02LjI1LTUuNjJhNS43IDUuNyAwIDAgMC00LjYgMi40NmMtMS0xLjI2LTEuOTgtMi4yNS0yLjg2LTIuODJBNy40IDcuNCAwIDAgMCAxNy41MiA0OFptMCA0Yy41MSAwIDEuMTQuMjIgMS44Mi42NSAyLjE0IDEuMzYgNi4yNSA4LjQzIDcuNzYgMTEuMTguNS45MiAxLjM3IDEuMzEgMi4xNCAxLjMxIDEuNTUgMCAyLjc1LTEuNTMuMTUtMy40OC0zLjkyLTIuOTMtMi41NS03LjcyLS42OC04LjAxLjA4LS4wMi4xNy0uMDIuMjQtLjAyIDEuNyAwIDIuNDUgMi45MyAyLjQ1IDIuOTNzMi4yIDUuNTIgNS45OCA5LjNjMy43NyAzLjc3IDMuOTcgNi44IDEuMjIgMTAuODMtMS44OCAyLjc1LTUuNDcgMy41OC05LjE2IDMuNTgtMy44MSAwLTcuNzMtLjktOS45Mi0xLjQ2LS4xMS0uMDMtMTMuNDUtMy44LTExLjc2LTcgLjI4LS41NC43NS0uNzYgMS4zNC0uNzYgMi4zOCAwIDYuNyAzLjU0IDguNTcgMy41NC40MSAwIC43LS4xNy44My0uNi43OS0yLjg1LTEyLjA2LTQuMDUtMTAuOTgtOC4xNy4yLS43My43MS0xLjAyIDEuNDQtMS4wMiAzLjE0IDAgMTAuMiA1LjUzIDExLjY4IDUuNTMuMTEgMCAuMi0uMDMuMjQtLjEuNzQtMS4yLjMzLTIuMDQtNC45LTUuMi01LjIxLTMuMTYtOC44OC01LjA2LTYuOC03LjMzLjI0LS4yNi41OC0uMzggMS0uMzggMy4xNyAwIDEwLjY2IDYuODIgMTAuNjYgNi44MnMyLjAyIDIuMSAzLjI1IDIuMWMuMjggMCAuNTItLjEuNjgtLjM4Ljg2LTEuNDYtOC4wNi04LjIyLTguNTYtMTEuMDEtLjM0LTEuOS4yNC0yLjg1IDEuMzEtMi44NVoiLz48cGF0aCBmaWxsPSIjRkZEMjFFIiBkPSJNMzguNiA3Ni42OWMyLjc1LTQuMDQgMi41NS03LjA3LTEuMjItMTAuODQtMy43OC0zLjc3LTUuOTgtOS4zLTUuOTgtOS4zcy0uODItMy4yLTIuNjktMi45Yy0xLjg3LjMtMy4yNCA1LjA4LjY4IDguMDEgMy45MSAyLjkzLS43OCA0LjkyLTIuMjkgMi4xNy0xLjUtMi43NS01LjYyLTkuODItNy43Ni0xMS4xOC0yLjEzLTEuMzUtMy42My0uNi0zLjEzIDIuMi41IDIuNzkgOS40MyA5LjU1IDguNTYgMTEtLjg3IDEuNDctMy45My0xLjcxLTMuOTMtMS43MXMtOS41Ny04LjcxLTExLjY2LTYuNDRjLTIuMDggMi4yNyAxLjU5IDQuMTcgNi44IDcuMzMgNS4yMyAzLjE2IDUuNjQgNCA0LjkgNS4yLS43NSAxLjItMTIuMjgtOC41My0xMy4zNi00LjQtMS4wOCA0LjExIDExLjc3IDUuMyAxMC45OCA4LjE1LS44IDIuODUtOS4wNi01LjM4LTEwLjc0LTIuMTgtMS43IDMuMjEgMTEuNjUgNi45OCAxMS43NiA3LjAxIDQuMyAxLjEyIDE1LjI1IDMuNDkgMTkuMDgtMi4xMloiLz48cGF0aCBmaWxsPSIjRkY5RDBCIiBkPSJNNzcuNCA0OGMxLjYyIDAgMy4wNy42NiA0LjA3IDEuODdhNS45NyA1Ljk3IDAgMCAxIDEuMzMgMy43NiA3LjEgNy4xIDAgMCAxIDEuOTUtLjNjMS41NSAwIDIuOTUuNTkgMy45NCAxLjY2YTUuOCA1LjggMCAwIDEgLjggNyA1LjMgNS4zIDAgMCAxIDEuNzggMi44MmMuMjQuOS40OCAyLjgtLjggNC43NGE1LjIyIDUuMjIgMCAwIDEgLjM3IDUuMDJjLTEuMDIgMi4zMi0zLjU3IDQuMTQtOC41MSA2LjEtMy4wOCAxLjIyLTUuOSAyLTUuOTIgMi4wMWE0NC4zMyA0NC4zMyAwIDAgMS0xMC45MyAxLjZjLTUuODYgMC0xMC4wNS0xLjgtMTIuNDYtNS4zNC0zLjg4LTUuNjktMy4zMy0xMC45IDEuNy0xNS45MiAyLjc4LTIuNzggNC42My02Ljg3IDUuMDEtNy43Ny43OC0yLjY2IDIuODMtNS42MiA2LjI0LTUuNjJhNS43IDUuNyAwIDAgMSA0LjYgMi40NmMxLTEuMjYgMS45OC0yLjI1IDIuODctMi44MkE3LjQgNy40IDAgMCAxIDc3LjQgNDhabTAgNGMtLjUxIDAtMS4xMy4yMi0xLjgyLjY1LTIuMTMgMS4zNi02LjI1IDguNDMtNy43NiAxMS4xOGEyLjQzIDIuNDMgMCAwIDEtMi4xNCAxLjMxYy0xLjU0IDAtMi43NS0xLjUzLS4xNC0zLjQ4IDMuOTEtMi45MyAyLjU0LTcuNzIuNjctOC4wMWExLjU0IDEuNTQgMCAwIDAtLjI0LS4wMmMtMS43IDAtMi40NSAyLjkzLTIuNDUgMi45M3MtMi4yIDUuNTItNS45NyA5LjNjLTMuNzggMy43Ny0zLjk4IDYuOC0xLjIyIDEwLjgzIDEuODcgMi43NSA1LjQ3IDMuNTggOS4xNSAzLjU4IDMuODIgMCA3LjczLS45IDkuOTMtMS40Ni4xLS4wMyAxMy40NS0zLjggMTEuNzYtNy0uMjktLjU0LS43NS0uNzYtMS4zNC0uNzYtMi4zOCAwLTYuNzEgMy41NC04LjU3IDMuNTQtLjQyIDAtLjcxLS4xNy0uODMtLjYtLjgtMi44NSAxMi4wNS00LjA1IDEwLjk3LTguMTctLjE5LS43My0uNy0xLjAyLTEuNDQtMS4wMi0zLjE0IDAtMTAuMiA1LjUzLTExLjY4IDUuNTMtLjEgMC0uMTktLjAzLS4yMy0uMS0uNzQtMS4yLS4zNC0yLjA0IDQuODgtNS4yIDUuMjMtMy4xNiA4LjktNS4wNiA2LjgtNy4zMy0uMjMtLjI2LS41Ny0uMzgtLjk4LS4zOC0zLjE4IDAtMTAuNjcgNi44Mi0xMC42NyA2Ljgycy0yLjAyIDIuMS0zLjI0IDIuMWEuNzQuNzQgMCAwIDEtLjY4LS4zOGMtLjg3LTEuNDYgOC4wNS04LjIyIDguNTUtMTEuMDEuMzQtMS45LS4yNC0yLjg1LTEuMzEtMi44NVoiLz48cGF0aCBmaWxsPSIjRkZEMjFFIiBkPSJNNTYuMzMgNzYuNjljLTIuNzUtNC4wNC0yLjU2LTcuMDcgMS4yMi0xMC44NCAzLjc3LTMuNzcgNS45Ny05LjMgNS45Ny05LjNzLjgyLTMuMiAyLjctMi45YzEuODYuMyAzLjIzIDUuMDgtLjY4IDguMDEtMy45MiAyLjkzLjc4IDQuOTIgMi4yOCAyLjE3IDEuNTEtMi43NSA1LjYzLTkuODIgNy43Ni0xMS4xOCAyLjEzLTEuMzUgMy42NC0uNiAzLjEzIDIuMi0uNSAyLjc5LTkuNDIgOS41NS04LjU1IDExIC44NiAxLjQ3IDMuOTItMS43MSAzLjkyLTEuNzFzOS41OC04LjcxIDExLjY2LTYuNDRjMi4wOCAyLjI3LTEuNTggNC4xNy02LjggNy4zMy01LjIzIDMuMTYtNS42MyA0LTQuOSA1LjIuNzUgMS4yIDEyLjI4LTguNTMgMTMuMzYtNC40IDEuMDggNC4xMS0xMS43NiA1LjMtMTAuOTcgOC4xNS44IDIuODUgOS4wNS01LjM4IDEwLjc0LTIuMTggMS42OSAzLjIxLTExLjY1IDYuOTgtMTEuNzYgNy4wMS00LjMxIDEuMTItMTUuMjYgMy40OS0xOS4wOC0yLjEyWiIvPjwvc3ZnPiA%3D" alt="VPM-Data"></a>
  <a href="https://huggingface.co/datasets/rooty2020/VP-WorldBench"><img src="https://img.shields.io/badge/VP--WorldBench-Benchmark-FFD21E.svg?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5NSIgaGVpZ2h0PSI4OCIgZmlsbD0ibm9uZSI%2BPHBhdGggZmlsbD0iI0ZGRDIxRSIgZD0iTTQ3LjIxIDc2LjVhMzQuNzUgMzQuNzUgMCAxIDAgMC02OS41IDM0Ljc1IDM0Ljc1IDAgMCAwIDAgNjkuNVoiLz48cGF0aCBmaWxsPSIjRkY5RDBCIiBkPSJNODEuOTYgNDEuNzVhMzQuNzUgMzQuNzUgMCAxIDAtNjkuNSAwIDM0Ljc1IDM0Ljc1IDAgMCAwIDY5LjUgMFptLTczLjUgMGEzOC43NSAzOC43NSAwIDEgMSA3Ny41IDAgMzguNzUgMzguNzUgMCAwIDEtNzcuNSAwWiIvPjxwYXRoIGZpbGw9IiMzQTNCNDUiIGQ9Ik01OC41IDMyLjNjMS4yOC40NCAxLjc4IDMuMDYgMy4wNyAyLjM4YTUgNSAwIDEgMC02Ljc2LTIuMDdjLjYxIDEuMTUgMi41NS0uNzIgMy43LS4zMlpNMzQuOTUgMzIuM2MtMS4yOC40NC0xLjc5IDMuMDYtMy4wNyAyLjM4YTUgNSAwIDEgMSA2Ljc2LTIuMDdjLS42MSAxLjE1LTIuNTYtLjcyLTMuNy0uMzJaIi8%2BPHBhdGggZmlsbD0iI0ZGMzIzRCIgZD0iTTQ2Ljk2IDU2LjI5YzkuODMgMCAxMy04Ljc2IDEzLTEzLjI2IDAtMi4zNC0xLjU3LTEuNi00LjA5LS4zNi0yLjMzIDEuMTUtNS40NiAyLjc0LTguOSAyLjc0LTcuMTkgMC0xMy02Ljg4LTEzLTIuMzhzMy4xNiAxMy4yNiAxMyAxMy4yNloiLz48cGF0aCBmaWxsPSIjM0EzQjQ1IiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0zOS40MyA1NGE4LjcgOC43IDAgMCAxIDUuMy00LjQ5Yy40LS4xMi44MS41NyAxLjI0IDEuMjguNC42OC44MiAxLjM3IDEuMjQgMS4zNy40NSAwIC45LS42OCAxLjMzLTEuMzUuNDUtLjcuODktMS4zOCAxLjMyLTEuMjVhOC42MSA4LjYxIDAgMCAxIDUgNC4xN2MzLjczLTIuOTQgNS4xLTcuNzQgNS4xLTEwLjcgMC0yLjM0LTEuNTctMS42LTQuMDktLjM2bC0uMTQuMDdjLTIuMzEgMS4xNS01LjM5IDIuNjctOC43NyAyLjY3cy02LjQ1LTEuNTItOC43Ny0yLjY3Yy0yLjYtMS4yOS00LjIzLTIuMS00LjIzLjI5IDAgMy4wNSAxLjQ2IDguMDYgNS40NyAxMC45N1oiIGNsaXAtcnVsZT0iZXZlbm9kZCIvPjxwYXRoIGZpbGw9IiNGRjlEMEIiIGQ9Ik03MC43MSAzN2EzLjI1IDMuMjUgMCAxIDAgMC02LjUgMy4yNSAzLjI1IDAgMCAwIDAgNi41Wk0yNC4yMSAzN2EzLjI1IDMuMjUgMCAxIDAgMC02LjUgMy4yNSAzLjI1IDAgMCAwIDAgNi41Wk0xNy41MiA0OGMtMS42MiAwLTMuMDYuNjYtNC4wNyAxLjg3YTUuOTcgNS45NyAwIDAgMC0xLjMzIDMuNzYgNy4xIDcuMSAwIDAgMC0xLjk0LS4zYy0xLjU1IDAtMi45NS41OS0zLjk0IDEuNjZhNS44IDUuOCAwIDAgMC0uOCA3IDUuMyA1LjMgMCAwIDAtMS43OSAyLjgyYy0uMjQuOS0uNDggMi44LjggNC43NGE1LjIyIDUuMjIgMCAwIDAtLjM3IDUuMDJjMS4wMiAyLjMyIDMuNTcgNC4xNCA4LjUyIDYuMSAzLjA3IDEuMjIgNS44OSAyIDUuOTEgMi4wMWE0NC4zMyA0NC4zMyAwIDAgMCAxMC45MyAxLjZjNS44NiAwIDEwLjA1LTEuOCAxMi40Ni01LjM0IDMuODgtNS42OSAzLjMzLTEwLjktMS43LTE1LjkyLTIuNzctMi43OC00LjYyLTYuODctNS03Ljc3LS43OC0yLjY2LTIuODQtNS42Mi02LjI1LTUuNjJhNS43IDUuNyAwIDAgMC00LjYgMi40NmMtMS0xLjI2LTEuOTgtMi4yNS0yLjg2LTIuODJBNy40IDcuNCAwIDAgMCAxNy41MiA0OFptMCA0Yy41MSAwIDEuMTQuMjIgMS44Mi42NSAyLjE0IDEuMzYgNi4yNSA4LjQzIDcuNzYgMTEuMTguNS45MiAxLjM3IDEuMzEgMi4xNCAxLjMxIDEuNTUgMCAyLjc1LTEuNTMuMTUtMy40OC0zLjkyLTIuOTMtMi41NS03LjcyLS42OC04LjAxLjA4LS4wMi4xNy0uMDIuMjQtLjAyIDEuNyAwIDIuNDUgMi45MyAyLjQ1IDIuOTNzMi4yIDUuNTIgNS45OCA5LjNjMy43NyAzLjc3IDMuOTcgNi44IDEuMjIgMTAuODMtMS44OCAyLjc1LTUuNDcgMy41OC05LjE2IDMuNTgtMy44MSAwLTcuNzMtLjktOS45Mi0xLjQ2LS4xMS0uMDMtMTMuNDUtMy44LTExLjc2LTcgLjI4LS41NC43NS0uNzYgMS4zNC0uNzYgMi4zOCAwIDYuNyAzLjU0IDguNTcgMy41NC40MSAwIC43LS4xNy44My0uNi43OS0yLjg1LTEyLjA2LTQuMDUtMTAuOTgtOC4xNy4yLS43My43MS0xLjAyIDEuNDQtMS4wMiAzLjE0IDAgMTAuMiA1LjUzIDExLjY4IDUuNTMuMTEgMCAuMi0uMDMuMjQtLjEuNzQtMS4yLjMzLTIuMDQtNC45LTUuMi01LjIxLTMuMTYtOC44OC01LjA2LTYuOC03LjMzLjI0LS4yNi41OC0uMzggMS0uMzggMy4xNyAwIDEwLjY2IDYuODIgMTAuNjYgNi44MnMyLjAyIDIuMSAzLjI1IDIuMWMuMjggMCAuNTItLjEuNjgtLjM4Ljg2LTEuNDYtOC4wNi04LjIyLTguNTYtMTEuMDEtLjM0LTEuOS4yNC0yLjg1IDEuMzEtMi44NVoiLz48cGF0aCBmaWxsPSIjRkZEMjFFIiBkPSJNMzguNiA3Ni42OWMyLjc1LTQuMDQgMi41NS03LjA3LTEuMjItMTAuODQtMy43OC0zLjc3LTUuOTgtOS4zLTUuOTgtOS4zcy0uODItMy4yLTIuNjktMi45Yy0xLjg3LjMtMy4yNCA1LjA4LjY4IDguMDEgMy45MSAyLjkzLS43OCA0LjkyLTIuMjkgMi4xNy0xLjUtMi43NS01LjYyLTkuODItNy43Ni0xMS4xOC0yLjEzLTEuMzUtMy42My0uNi0zLjEzIDIuMi41IDIuNzkgOS40MyA5LjU1IDguNTYgMTEtLjg3IDEuNDctMy45My0xLjcxLTMuOTMtMS43MXMtOS41Ny04LjcxLTExLjY2LTYuNDRjLTIuMDggMi4yNyAxLjU5IDQuMTcgNi44IDcuMzMgNS4yMyAzLjE2IDUuNjQgNCA0LjkgNS4yLS43NSAxLjItMTIuMjgtOC41My0xMy4zNi00LjQtMS4wOCA0LjExIDExLjc3IDUuMyAxMC45OCA4LjE1LS44IDIuODUtOS4wNi01LjM4LTEwLjc0LTIuMTgtMS43IDMuMjEgMTEuNjUgNi45OCAxMS43NiA3LjAxIDQuMyAxLjEyIDE1LjI1IDMuNDkgMTkuMDgtMi4xMloiLz48cGF0aCBmaWxsPSIjRkY5RDBCIiBkPSJNNzcuNCA0OGMxLjYyIDAgMy4wNy42NiA0LjA3IDEuODdhNS45NyA1Ljk3IDAgMCAxIDEuMzMgMy43NiA3LjEgNy4xIDAgMCAxIDEuOTUtLjNjMS41NSAwIDIuOTUuNTkgMy45NCAxLjY2YTUuOCA1LjggMCAwIDEgLjggNyA1LjMgNS4zIDAgMCAxIDEuNzggMi44MmMuMjQuOS40OCAyLjgtLjggNC43NGE1LjIyIDUuMjIgMCAwIDEgLjM3IDUuMDJjLTEuMDIgMi4zMi0zLjU3IDQuMTQtOC41MSA2LjEtMy4wOCAxLjIyLTUuOSAyLTUuOTIgMi4wMWE0NC4zMyA0NC4zMyAwIDAgMS0xMC45MyAxLjZjLTUuODYgMC0xMC4wNS0xLjgtMTIuNDYtNS4zNC0zLjg4LTUuNjktMy4zMy0xMC45IDEuNy0xNS45MiAyLjc4LTIuNzggNC42My02Ljg3IDUuMDEtNy43Ny43OC0yLjY2IDIuODMtNS42MiA2LjI0LTUuNjJhNS43IDUuNyAwIDAgMSA0LjYgMi40NmMxLTEuMjYgMS45OC0yLjI1IDIuODctMi44MkE3LjQgNy40IDAgMCAxIDc3LjQgNDhabTAgNGMtLjUxIDAtMS4xMy4yMi0xLjgyLjY1LTIuMTMgMS4zNi02LjI1IDguNDMtNy43NiAxMS4xOGEyLjQzIDIuNDMgMCAwIDEtMi4xNCAxLjMxYy0xLjU0IDAtMi43NS0xLjUzLS4xNC0zLjQ4IDMuOTEtMi45MyAyLjU0LTcuNzIuNjctOC4wMWExLjU0IDEuNTQgMCAwIDAtLjI0LS4wMmMtMS43IDAtMi40NSAyLjkzLTIuNDUgMi45M3MtMi4yIDUuNTItNS45NyA5LjNjLTMuNzggMy43Ny0zLjk4IDYuOC0xLjIyIDEwLjgzIDEuODcgMi43NSA1LjQ3IDMuNTggOS4xNSAzLjU4IDMuODIgMCA3LjczLS45IDkuOTMtMS40Ni4xLS4wMyAxMy40NS0zLjggMTEuNzYtNy0uMjktLjU0LS43NS0uNzYtMS4zNC0uNzYtMi4zOCAwLTYuNzEgMy41NC04LjU3IDMuNTQtLjQyIDAtLjcxLS4xNy0uODMtLjYtLjgtMi44NSAxMi4wNS00LjA1IDEwLjk3LTguMTctLjE5LS43My0uNy0xLjAyLTEuNDQtMS4wMi0zLjE0IDAtMTAuMiA1LjUzLTExLjY4IDUuNTMtLjEgMC0uMTktLjAzLS4yMy0uMS0uNzQtMS4yLS4zNC0yLjA0IDQuODgtNS4yIDUuMjMtMy4xNiA4LjktNS4wNiA2LjgtNy4zMy0uMjMtLjI2LS41Ny0uMzgtLjk4LS4zOC0zLjE4IDAtMTAuNjcgNi44Mi0xMC42NyA2Ljgycy0yLjAyIDIuMS0zLjI0IDIuMWEuNzQuNzQgMCAwIDEtLjY4LS4zOGMtLjg3LTEuNDYgOC4wNS04LjIyIDguNTUtMTEuMDEuMzQtMS45LS4yNC0yLjg1LTEuMzEtMi44NVoiLz48cGF0aCBmaWxsPSIjRkZEMjFFIiBkPSJNNTYuMzMgNzYuNjljLTIuNzUtNC4wNC0yLjU2LTcuMDcgMS4yMi0xMC44NCAzLjc3LTMuNzcgNS45Ny05LjMgNS45Ny05LjNzLjgyLTMuMiAyLjctMi45YzEuODYuMyAzLjIzIDUuMDgtLjY4IDguMDEtMy45MiAyLjkzLjc4IDQuOTIgMi4yOCAyLjE3IDEuNTEtMi43NSA1LjYzLTkuODIgNy43Ni0xMS4xOCAyLjEzLTEuMzUgMy42NC0uNiAzLjEzIDIuMi0uNSAyLjc5LTkuNDIgOS41NS04LjU1IDExIC44NiAxLjQ3IDMuOTItMS43MSAzLjkyLTEuNzFzOS41OC04LjcxIDExLjY2LTYuNDRjMi4wOCAyLjI3LTEuNTggNC4xNy02LjggNy4zMy01LjIzIDMuMTYtNS42MyA0LTQuOSA1LjIuNzUgMS4yIDEyLjI4LTguNTMgMTMuMzYtNC40IDEuMDggNC4xMS0xMS43NiA1LjMtMTAuOTcgOC4xNS44IDIuODUgOS4wNS01LjM4IDEwLjc0LTIuMTggMS42OSAzLjIxLTExLjY1IDYuOTgtMTEuNzYgNy4wMS00LjMxIDEuMTItMTUuMjYgMy40OS0xOS4wOC0yLjEyWiIvPjwvc3ZnPiA%3D" alt="VP-WorldBench"></a>
</p>

<div align="center">
  <a href="https://rootyjeon.github.io/">Byungwoo&nbsp;Jeon</a><sup>1*</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://sunghwanhong.github.io/">Sunghwan&nbsp;Hong</a><sup>2*</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://openreview.net/profile?id=~Dohyeon_Kim5">Dohyeon&nbsp;Kim</a><sup>3</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://jhoonjwa.github.io/">Jonghoon&nbsp;Lee</a><sup>1</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://scholar.google.com/citations?user=k2e82kcAAAAJ&hl=en">Jungwoo&nbsp;Park</a><sup>1</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://blog.sulwon.com/">Inhee&nbsp;Lee</a><sup>3</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://pslab.ai/en/">Doosan&nbsp;Baek</a><sup>6</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://pslab.ai/en/">Jinil&nbsp;Kim</a><sup>6</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://pslab.ai/en/">Taehyun&nbsp;Lim</a><sup>6</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://jhugestar.github.io/">Hanbyul&nbsp;Joo</a><sup>3,5</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://people.inf.ethz.ch/pomarc/">Marc&nbsp;Pollefeys</a><sup>2</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://www.sainingxie.com/">Saining&nbsp;Xie</a><sup>4&dagger;</sup> &ensp;<b>&middot;</b>&ensp;
  <a href="https://alinlab.kaist.ac.kr/shin.html">Jinwoo&nbsp;Shin</a><sup>1,5&dagger;</sup>
  <br>
  <br>
  <sup>1</sup>KAIST &emsp; <sup>2</sup>ETH Zürich &emsp; <sup>3</sup>Seoul National University &emsp; <sup>4</sup>New York University &emsp; <sup>5</sup>RLWRLD &emsp; <sup>6</sup>Physics Sim Lab
  <br>
  <sup>*</sup>Equal Contribution &emsp; <sup>&dagger;</sup>Equal Advising &emsp;<br>
</div>

<h3 align="center">[<a href="https://rootyjeon.github.io/video-point-model/">project page</a>]</h3>

<br>

**Video-Point Model (VPM)** is a multi-view video world model built on [NVIDIA Cosmos3](https://huggingface.co/nvidia/Cosmos3-Nano).

- 🎥 **Multi-view Video**: Generating synchronized videos from multi-cameras.
- 📍 **3D Point Tracks**: Consistent 3D points across views, read from its own attention features.
- 🦾 **Robot action**: Predicting a robot action chunk together with the future frames.

## 🔧 Installation

```bash
git clone https://github.com/vpm-project/VPM.git && cd VPM
sudo apt-get install -y ffmpeg
uv sync --extra train --group cu130-train      # CUDA 12.8: --group cu128-train
source .venv/bin/activate
```

## 📊 VP-WorldBench

VP-WorldBench is a 4D world modeling benchmark for video world models: 1,200 synchronized multi-view clips with cameras,
3D point tracks and captions, in four subsets (simulated / real × robot / general). Its main task is multi-view
video generation, scored on appearance, cross-view consistency and 3D motion accuracy. VP-WorldBench can be used to evaluate diverse tasks such as 3D/4D reconstruction, human motion, and video understanding.

```bash
pip install -e "vp-worldbench[metrics,fvd]"
vpmwb generation export   --root data/VP-WorldBench --out inputs/
vpmwb generation evaluate --root data/VP-WorldBench --pred preds/my_model --out runs/my_model
```

Details for download, results, and the evaluation on other tasks are in [vp-worldbench/README.md](vp-worldbench/README.md).

## 🗂️ VPM-Data

VPM-Data is the training set of VPM: **10K simulated clips** that pair synchronized multi-view videos with persistent
3D point trajectories. It has 8,000 tabletop manipulation clips, in which a Fourier GR-1 T2 humanoid picks and places
objects among clutter, and 2,000 urban clips of pedestrians and vehicles filmed by moving cameras.

Every clip comes with

- 🎥 **multi-view videos**, time-synchronized across the cameras;
- 📷 **camera intrinsics and extrinsics** for every view and frame;
- 📍 **persistent 3D point trajectories** on the robot, the manipulated objects, people and vehicles, with per-view visibility;
- 🌊 **depth maps** rendered for every view and frame;
- 📝 **language captions**;
- 🏷️ **metadata**, such as object and actor labels, target visibility states (visible, occluded, out of frame),
  camera motion types, and scene and task attributes.

All annotations come from the simulator, so they are exact and aligned across views and time. This makes VPM-Data
useful well beyond world modeling:

| research task | annotations used |
|:--|:--|
| Multi-view and camera-controlled video generation | videos, cameras, captions |
| 3D/4D reconstruction and novel view synthesis | videos, cameras, depth |
| Depth estimation (monocular, multi-view, video) | videos, depth |
| Camera pose estimation | videos, cameras |
| 2D/3D and multi-view point tracking, scene flow | 3D trajectories, visibility |
| Human motion estimation and forecasting | 3D trajectories on people, cameras |
| Object and vehicle motion estimation | 3D trajectories on objects and vehicles, cameras |
| Robot manipulation and hand–object interaction | robot and object trajectories, captions |
| Occlusion reasoning and object permanence | visibility, depth |
| Video understanding (captioning, grounding, video QA) | videos, captions, metadata |

The files and their formats are described on the [🤗 dataset card](https://huggingface.co/datasets/rooty2020/VPM-Data).

## 🦾 Robot Data with 3D Point Trajectories

We also release the robot manipulation data with 3D point trajectories that we built for VPM. Every frame comes with
persistent, metric 3D points on the robot arm and gripper (and the manipulated objects in simulation), computed from
the robot CAD model and the recorded robot or simulator states instead of a visual point tracker, together with
calibrated multi-view videos and language. DROID-VP keeps whole DROID episodes with the calibration and per-frame
poses of all three cameras (the data VPM-DROID is trained on), and RoboCasa episodes are cut into language-annotated
subtask clips. The curation is described on each dataset card and in Appendix B.2 of the paper.

| dataset | source | samples | views | 3D points per frame | size | download |
|:--|:--|:--|:--|:--|:--|:--|
| DROID-VP | [DROID](https://droid-dataset.github.io/) (real) | 50,430 episodes, 17,982 with point tracks | 3 × 320×180, 15 fps | 512 arm + 256 gripper | 1.03 TB | [🤗 DROID-VP](https://huggingface.co/datasets/rooty2020/DROID-VP) |
| LIBERO-VP | [LIBERO](https://libero-project.github.io/) (sim) | 6,500 episodes, 130 tasks | 3 × 256×256, 20 fps | 512 arm + 256 gripper + 256 object | 13.8 GB | [🤗 LIBERO-VP](https://huggingface.co/datasets/rooty2020/LIBERO-VP) |
| RoboCasa-VP | [RoboCasa365](https://robocasa.ai/) (sim) | 217,594 subtask clips from 24,504 episodes, 235 tasks | 3 × 256×256, 20 fps | 512 arm + 256 gripper + 256 per object | 487 GB | [🤗 RoboCasa-VP](https://huggingface.co/datasets/rooty2020/RoboCasa-VP) |

```bash
hf download rooty2020/DROID-VP    --repo-type dataset --local-dir data/DROID-VP
hf download rooty2020/LIBERO-VP   --repo-type dataset --local-dir data/LIBERO-VP
hf download rooty2020/RoboCasa-VP --repo-type dataset --local-dir data/RoboCasa-VP
```

LIBERO-VP and RoboCasa-VP ship as tar shards of self-contained samples (videos, `robot.npz` / `gripper.npz` /
`object*.npz` point tracks and a `meta.json` with cameras and language). DROID-VP keeps the layout of the VPM-DROID
training run (JPEG frame caches, joint / pose / camera tables and point caches) and comes with a reader,
`tools/read_episode.py`. To train on them, convert them to the slot layout in
[training/DATA.md](training/DATA.md#preparing-your-own-data). They follow the licenses of their source datasets
(CC BY 4.0).

## 📦 Checkpoints

```bash
bash scripts/download_assets.sh             # Cosmos3-Nano, Wan2.2 VAE and the Qwen3-VL tokenizer
bash scripts/download_assets.sh --release   # + the VPM-DROID checkpoint
```

Cosmos3-Nano is converted to a DCP checkpoint for training, as in
[Cosmos-Framework](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md#step-2--prepare-checkpoint).

| model | setup | download |
|:--|:--|:--|
| VPM-DROID | Video + Action + Point tracks, three DROID cameras incl. wrist | [🤗 VPM-DROID](https://huggingface.co/rooty2020/VPM-DROID) |
| VPM-DROID-v2 | Video + Action, three DROID cameras incl. wrist, trained on [DROID-VP](https://huggingface.co/datasets/rooty2020/DROID-VP) | [🤗 VPM-DROID-v2](https://huggingface.co/rooty2020/VPM-DROID-v2) |

## 🚀 Inference

VPM-DROID takes all three DROID cameras (exterior left, exterior right, wrist) and generates the next 16 frames of
every camera, a 16-step action chunk and, with `--tracks grid`, point tracks, from the last 9 frames and the actions
between them:

```bash
bash inference/infer_droid.sh \
    --past-frames past.npz --past-actions past_actions.npy \
    --state q0,q1,q2,q3,q4,q5,q6,gripper --prompt "put the marker in the cup"
```

Outputs (videos, `actions.npz`, optional point tracks) are written to `outputs/droid_inference/`.

Input formats, options and multi-GPU use are in [inference/README.md](inference/README.md).

## 🏋️ Training

VPM is trained from Cosmos3-Nano in three stages:

| stage | starts from | trains |
|:--|:--|:--|
| **1**  | Cosmos3-Nano | video generator, camera conditioning |
| **2**  | stage 1 | tracking head |
| **3**  | stage 2 | video generator, action generator, tracking head |

```bash
bash scripts/download_assets.sh --data all   # VPM-Data
bash training/train_stage1.sh
bash training/train_stage2.sh
bash training/train_stage3.sh
```

Settings, multi-node runs, resuming and checkpoint export are in [training/README.md](training/README.md);
the data format is in [training/DATA.md](training/DATA.md).

## 🔗 Related Projects

- [Cosmos-Framework](https://github.com/NVIDIA/cosmos-framework) and [Cosmos 3](https://huggingface.co/collections/nvidia/cosmos3): the training framework and base model VPM builds on.
- [MVTracker](https://github.com/ethz-vlg/mvtracker): multi-view 3D point tracking, used by VP-WorldBench.
- [Wan2.2](https://github.com/Wan-Video/Wan2.2): the video VAE.
- [DROID](https://droid-dataset.github.io/): the robot manipulation dataset.

## 📜 License

The code and the VPM model weights are released under [CC BY-NC 4.0](LICENSE) (attribution: **Byungwoo Jeon**).
Non-commercial use only. For commercial licensing contact imbw2024@kaist.ac.kr.

**Built on NVIDIA Cosmos.** VPM is a derivative of [cosmos-framework](https://github.com/NVIDIA/cosmos-framework) and
the base model [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) (Copyright (c) 2025-2026 NVIDIA
CORPORATION & AFFILIATES), which NVIDIA distributes under the OpenMDW License Agreement 1.1. As that agreement
requires, a copy is included as [LICENSE-OpenMDW-1.1.txt](LICENSE-OpenMDW-1.1.txt) and NVIDIA's copyright notices are
retained. Its terms apply to the Cosmos-derived portions, and the CC BY-NC 4.0 terms apply to VPM as released.
Other third-party code is listed in [NOTICE](NOTICE). Datasets follow the licenses on their Hugging Face cards.

## 📝 Citation

```bibtex
@misc{vpm2026,
  title  = {Learning the 4D World through Point Trajectory Forecasting},
  author = {Jeon, Byungwoo and Hong, Sunghwan and Kim, Dohyeon and Lee, Jonghoon and Park, Jungwoo and Lee, Inhee and
            Baek, Doosan and Kim, Jinil and Lim, Taehyun and Joo, Hanbyul and Pollefeys, Marc and Xie, Saining and
            Shin, Jinwoo},
  year   = {2026},
  note   = {Preprint},
  url    = {https://rootyjeon.github.io/video-point-model/}
}
```
