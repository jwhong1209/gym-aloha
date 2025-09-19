
# Mobile ALOHA MJCF 파일 작성
## CAD 분석 
>[!warning] CAD 데이터와 MuJoCo 좌표계 기준이 다르니 주의

![[Pasted image 20250917204647.png]]
- "mobile base 원점 - WindowX AI arm base 원점" : (337, 324.5, 1043.16) mm


![[Pasted image 20250918091625.png]]
- STL 파일 내보낼 때 원점이 이상하게 잡히는 이슈 → 솔리드웍스에서 흔히 발생하는 이슈.

![[Pasted image 20250918094317.png]]
좌표계를 새로 생성하고,

![[Pasted image 20250918094341.png]]
STL로 내보낼 때 좌표계 설정 및 옵션 - STL 출력 데이터를 번역하지 않음 설정
![[Pasted image 20250918094406.png]]

그러면 원점 및 좌표계가 의도한대로 설정된다.
![[Pasted image 20250918094433.png]]


>[!quote]
>https://www.reddit.com/r/SolidWorks/comments/j5gz2m/saving_as_stls_shifted_my_geometry/?tl=ko

Assembly 파일을 STL로 그대로 변환하면 파트 별로 STL 파일 따로 만들어짐 → MJCF 또는 URDF 파일로 만들기 까다로움. (그럴 필요도 없고)
전체 assembly파일을 part 파일로 저장 후 STL 파일로 변환
