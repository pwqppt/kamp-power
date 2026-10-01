# 로컬 PC와 GitHub로 옮기기

현재 작성 환경의 `/workspace/scratch/791784fae68e/kamp_power`는 사용자 Windows PC 경로가 아닙니다.
권장 PC 경로: `%USERPROFILE%\Projects\kamp-power` (실제 경로는 아래 PowerShell 명령이 표시합니다).

1. 제공 ZIP을 다운로드하여 압축 해제합니다. 내부 `kamp_power` 폴더를 위 `kamp-power` 이름으로 옮깁니다.
2. Git와 GitHub CLI(gh)가 설치된 PowerShell에서 아래 명령을 실행합니다. 기준 저장소는 `https://github.com/pwqppt/kamp-power`입니다.

```powershell
$projectPath = Join-Path $env:USERPROFILE 'Projects\kamp-power'
Set-Location $projectPath
(Get-Location).Path
gh auth login --hostname github.com --git-protocol https --web
gh api user --jq .login
# 출력 계정이 pwqppt인지 확인한 뒤 진행
git init -b main
git add .
git -c user.name='Project checkpoint' -c user.email='checkpoint@users.noreply.github.com' commit -m 'Checkpoint: data audit, forecast experiments and diagnostics'
git remote add origin https://github.com/pwqppt/kamp-power.git
git push -u origin main --follow-tags
```

이미 같은 이름의 저장소가 있으면 create를 반복하지 말고 URL과 내용을 확인하세요. 계정이 다르면 업로드하지 마세요. 원본 데이터가 포함되므로 비공개를 유지하고, 외부 공개 전 배포 권한을 확인하세요. 이메일 주소만으로 GitHub 로그인이나 저장소 생성은 할 수 없습니다.

다른 PC:
```bash
gh auth login
gh repo clone pwqppt/kamp-power
cd kamp-power
```

이후 README의 Python 환경 설정을 따릅니다. 다른 ChatGPT 계정에서도 해당 비공개 저장소에 대한 별도 접근 연결 또는 프로젝트 ZIP 업로드가 필요합니다. GitHub 저장소는 대화 이력을 자동 전송하지 않으며 HANDOFF.md가 연구 맥락을 전달합니다.

원본 ZIP 전체에는 불필요한 다른 데이터 ZIP이 섞여 있어, 저장소에는 실제 사용한 원본 CSV를 그대로 포함합니다. SHA256으로 일치 여부를 확인할 수 있습니다. `.venv`, 임시 캐시, 재생성 가능한 features.pkl은 Git에서 제외합니다.
