import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const root=path.resolve(process.env.RESEARCH_ROOT || '../..');
const skill=process.env.PRESENTATION_SKILL_ROOT;
const runtime=process.env.ARTIFACT_PYTHON;
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const facts=JSON.parse(await fs.readFile(path.join(root,'outputs/submission/facts.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Malgun Gothic',navy='#173B4F',blue='#1D647F',orange='#C87D3B',gray='#8A96A3',red='#B34C3B';
function text(s,t,x,y,w,h,size=28,bold=false,color=navy){const o=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});o.text=t;o.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};return o;}
function slide(title,note){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,64,44,1152,104,40,true);text(s,`${p.slides.items.length}   ${note}`,64,668,1152,30,16,false,'#5A6D78');s.speakerNotes.textFrame.setText(note);return s;}
function table(s,values,x=64,y=170,w=1152,h=350,widths){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const z=t.getCell(r,c);z.text.style={typeface:font,fontSize:24,color:r===0?'#FFFFFF':navy,bold:r===0};z.fill=r===0?navy:(r%2?'#F0F4F6':'#FFFFFF');}t.borders.assign({fill:'#D8E1E7',width:1,style:'solid'});return t;}
function chart(s,cats,values,x,y,w,h,title){const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories:cats,series:[{name:title,values,fill:blue,dataLabelOverrides:values.map((v,idx)=>({idx,text:v.toFixed(2),textStyle:{typeface:font,fontSize:20}}))}],barOptions:{direction:'column',grouping:'clustered'},hasLegend:false,dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:20,typeface:font}},xAxis:{textStyle:{fontSize:20,typeface:font}},yAxis:{min:0,max:25,majorUnit:5,textStyle:{fontSize:20,typeface:font}},title,titleTextStyle:{fontSize:24,typeface:font}});applyPresentationChartFont(c,{fontFamily:font});return c;}
function para(s,items,y=180){items.forEach((a,i)=>text(s,a,78,y+i*112,1110,98,28));}

let s=slide('평균 전력과 일최대 위험을 함께 예측하다','과제 ⑤ | 일반국민·대학(원)생 부문 | 연구 초안');
text(s,'생산기록 문맥을 활용한 15분 예측과\n별도 일최대 추정',70,186,1120,130,48,true);
text(s,'4~6월 평균 MAE 11.4% 감소\n별도 일최대 MAE 6.1% 감소',74,386,1050,126,38,true,blue);
text(s,'전주 기준 대비 관측값 · 새 독립기간의 우위는 미확정',74,548,1120,52,24,false);

s=slide('공식 과제의 세 질문에 답한다','출처: KAMP 2026-09-22 수정 공지, 과제⑤ 및 서면평가표');
para(s,['1. 다음 날 전력사용량을 얼마나 잘 예측하는가?','2. 언제 오차가 크고, 언제 최대피크가 발생하는가?','3. 생산 제약을 지키며 부하를 옮기면 어떤 결과가 나는가?']);
text(s,'평가 배점 15 / 40 / 15 / 10 / 10 / 10점에 맞춰 증거를 연결',78,557,1120,60,26,true,blue);
s.speakerNotes.textFrame.setText('공식 기준: https://www.kamp-ai.kr/contestNoticeDetail?CPT_NOTICE_SEQ=28 . 보고서 6개 장과 docs/EVALUATION_TRACEABILITY.md 참조.');

s=slide('데이터의 반복과 계측 한계를 먼저 확인했다','증거: outputs/data_audit.json, research_v4/pattern_overlap.csv');
table(s,[['계측 범위','검증한 사실','해석의 한계'],['257일 / 6,168행','24,672개 15분 구간','전력 단위는 미확인'],['142개 고유 일패턴','160일이 반복 그룹에 포함','독립 표본 가정에 주의'],['7/13·7/15 시간 오류','2일 정답·전력 이력 제외','행 순서 복원은 가정'],['생산·공장인원 기록','시간 단위, 인원 소수값 존재','설비 가동·이동 가능성 미확인']],64,160,1152,355,[280,410,462]);
text(s,'공식 재다운로드 CSV와 원자료 SHA256 일치',78,554,1120,68,28,true,blue);

s=slide('D-1 16시 이후의 실측 정보는 입력하지 않는다','증거: check_peak_study.py · 미래 원자료 교란 시 특징 불변');
table(s,[['사용 가능','사용 금지'],['이미 끝난 전력 구간과 과거 생산 문맥','다음 날 실측 기온·생산량·전력'],['달력 / 2·7·14일 lag / 최근 통계','검증일 실제 생산상태로 전문가 선택'],['4~6월 월별 expanding validation','7월 결과에 맞춘 설정 재조정']],64,170,1152,300,[576,576]);
text(s,'검증 91일 / 8,736구간 · 7월은 사후 진단 29일\n8~9월은 이미 사용된 역사적 테스트이며 새 독립 검증이 아님',78,510,1120,120,27,false);

s=slide('같은 시간 조건에서 모델과 피처를 비교했다','출처: outputs/research_v4/benchmark_scores.csv · 원자료 단위');
const bm=Object.fromEntries(facts.benchmark.map(r=>[r.candidate,r]));
table(s,[['모델','평균 MAE','RMSE','곡선 일최대 MAE'],...['naive','Ridge_F2','ExtraTrees_F2','LGB_F2','Mixture_B1','reference'].map((k,i)=>[['전주 기준','Ridge','ExtraTrees','LGB 단일','상태 혼합','최종 구간 결합'][i],...['mae','rmse','daily_peak_mae'].map(c=>bm[k][c].toFixed(3))])],64,156,1152,414,[354,240,240,318]);
text(s,'과거 문맥과 상태 결합이 평균 예측에 기여했다',78,595,1120,50,27,true,blue);

s=slide('평균 곡선과 일최대 추정의 역할을 분리했다','4~6월 91일 · 평균곡선은 유지 · H6 고정 0.5 결합');
chart(s,['전주 기준','최종 구간 결합'],[19.083333,16.910845],55,165,556,350,'15분 평균 MAE');
chart(s,['전주 기준','곡선 최대','별도 최대'],[18.923077,19.992784,17.771216],650,165,570,350,'일최대 MAE');
text(s,'별도 일최대 = (결합 곡선 최대 + 전주 곡선 최대) / 2',78,550,1120,54,29,true,blue);
text(s,'96구간 곡선의 최대값과 별도 출력은 서로 다를 수 있음',78,606,1120,45,22);

s=slide('피크 탐지는 개선됐지만 전주 기준과 교환관계가 있다','출처: outputs/peak_v4/H6/scores.csv · 4~6월');
table(s,[['일 단위 지표','전주 기준','결합 곡선 최대','별도 일최대'],['재현율','72.7%','29.5%','52.3%'],['정밀도','69.6%','68.4%','76.7%'],['F1','0.711','0.413','0.622'],['미탐 / 오경보','12 / 14','31 / 6','21 / 7']],64,168,1152,340,[336,272,272,272]);
text(s,'4~6월 일최대 MAE 차이(별도-전주): -1.152\n95% bootstrap 구간 [-3.778, 0.991] · 통계적 우위 확정 불가',78,540,1120,110,26);

s=slide('7월: 평균오차 감소와 피크 탐지를 함께 확인했다','7/13·7/15 제외 29일 · 사후 진단, 설정 변경 없음');
table(s,[['7월 지표','전주 기준','최종 구성'],['15분 MAE','12.177','10.552'],['15분 RMSE','26.381','17.007'],['일최대 MAE','16.621','17.483 (별도 출력)'],['피크일 재현율','95.0%','95.0% (별도 출력)'],['구간 피크 재현율','63.8%','36.4% (곡선)']],64,165,1152,390,[432,360,360]);
text(s,'별도 출력은 기존 결합의 일최대 MAE 20.438을 낮췄다',78,585,1120,60,27,true,blue);

s=slide('7월의 고온·생산기록·시간대가 오류와 함께 변했다','사후 연관성 분석 · 실제 미래 기온·생산량은 입력에 미사용');
table(s,[['조건','표본','관측 결과'],['30°C 초과','256구간 / 12일','MAE16.153, 편향 -11.090'],['생산기록 양수','1,764구간 / 24일','실제 피크442개 중437개'],['10~17시','812구간 / 29일','피크297개 중197개 미탐'],['최근 기온 +2°C 초과','단1일','효과를 일반화할 근거 부족']],64,165,1152,338,[338,352,462]);
text(s,'냉방 영향은 가설로 유지한다\n요일·생산상태·계절의 혼재 때문에 인과를 단정하지 않는다',78,537,1120,105,29,true,blue);

s=slide('시간 이동은 총량을 보존해도 피크를 악화시킬 수 있다','고정 이동비율10% / ±60분 · 실제 운영 검증이 아닌 조건부 계산');
table(s,[['평가','4~6월','7월'],['평균 일최대 감소','4.154','6.357'],['피크 악화일','16 / 91일','4 / 29일'],['기간 최대값 감소','-21.273','12.342']],64,170,1152,260,[432,360,360]);
text(s,'전력과 생산 장부에 같은 이동행렬을 적용하면 총량은 보존\n설비 제약·납기·생산 실현 가능성은 데이터로 확인할 수 없음',78,477,1120,120,29);

s=slide('4월 2일: 예측이 낮은 시간으로 옮기며 새 피크를 만들었다','출처: outputs/peak_v3/simulation/example_series.csv.gz · 원자료 단위');
text(s,'실제 일최대 181 → 243.273',78,180,1120,95,50,true,red);
para(s,['평균 저감량만으로 자동 일정 변경을 권고하지 않는다.','실시간 계측과 설비별 이동 가능성·생산 제약이 필요하다.','작업자 점검 우선순위를 제공하고 현장 파일럿에서 확인한다.'],308);

s=slide('요금 절감은 계약의 과거 최대수요와 인건비에 좌우된다','2026 산업용(을) 고압A 선택I를 2021 부하에 가상 적용 · 실제 청구액 아님');
table(s,[['완전월 전기요금 감소','잔존200kW 가정','잔존250kW 가정'],['4월','-308,794원','3,637원'],['5월','3,298원','3,298원'],['6월','165,429원','10,657원']],64,162,1152,265,[432,360,360]);
text(s,'2명 × 0.5시간 × 30일을 주간에서 야간으로 옮기면\n추가가산 195,000~240,000원 → 6월 절감보다 큼',78,460,1120,120,30,true,blue);
text(s,'kW·계약300kW·노동조건은 가정. 7월은 결측일로 월 청구액 미산정.',78,608,1120,44,21);
s.speakerNotes.textFrame.setText('공식 요금: 기후에너지환경부·한전 2026-03-13 붙임2 8쪽 https://www.mcee.go.kr/home/file/readDownloadFile.do?fileId=319782&fileSeq=2 . 근로기준법56조 https://law.go.kr/LSW/lsLinkCommonInfo.do?chrClsCd=010202&lsJoLnkSeq=1007688483 . 경제성은 사후 조건부 비교.');

s=slide('검증된 기여와 현장 적용 조건을 함께 제시한다','코드·전체 지표·실패·재현 명령은 제출 ZIP에 포함');
para(s,['기여: 전주 기준 대비 평균 MAE11.4%, 별도 일최대 MAE6.1% 감소','한계: 반복자료·검증 재사용·피크 재현율 교환관계·불확실성','현장 조건: 계측단위·설비 제약·생산 유지·계약·독립 검증 확인']);
text(s,'재현: python -X utf8 run_submission.py all\n제출 전: 실제 설문 완료 화면, 팀 정보와 서명 보완',78,540,1120,106,25,true,blue);

await fs.mkdir(path.join(root,'work/deck/reviewed'),{recursive:true});
const candidate=path.join(root,'work/deck/candidate-reviewed.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const final=path.join(root,'outputs/submission/presentation_reviewed.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,pythonExecutable:runtime,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,4,5,7,8,9,10,12].flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:[3,4,5,7,8,9,10,12],requiredNativeChartOwnerSlides:[6],fontPolicy:{basis:'design',families:[font]},
 materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,receiptPath:path.join(root,'work/deck/validation-reviewed.json')});
console.log(JSON.stringify(result));
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<finalDeck.slides.items.length;i++){
 const slide=finalDeck.slides.items[i];const b=await finalDeck.export({slide,format:'png',scale:1.5});await fs.writeFile(path.join(root,`work/deck/reviewed/slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
console.log('Rendered',finalDeck.slides.items.length,'slides');

