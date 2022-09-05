


function pullData(path,callback){
	var xhttp = new XMLHttpRequest();
	xhttp.open("GET",path,true);
	xhttp.onreadystatechange=function(){
		if (this.readyState == 4 && this.status == 200) {
			callback(this);
		};
	};
	xhttp.send()
}

var xmlDoc;
function initialLoadData(xml){
	xmlDoc=xml.responseXML
	/* console.log(xmlDoc) */
	
	loadData(String(01),1)

	$(".btn-minimize").click(function(){
		$(this).toggleClass('btn-plus');
		$(".minimizeContent").toggle();
	  });
	
	/*can run load dropdown here*/
	
/* 	console.log(xmlDoc)
	 */
	let playerNodes=xmlDoc.evaluate("//Name",xmlDoc,null,XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,null);
	/* let players=playerNodes.singleNodeValue; */
	
	/* console.log(playerNodes.snapshotLength) */
	
	for (let p_ID=1;p_ID<(playerNodes.snapshotLength+1);p_ID++){
		
		let dropName=document.getElementById('Player'+String(p_ID))
		
		dropName.innerHTML=playerNodes.snapshotItem(p_ID-1).textContent
		
	}

};

pullData("https://sagesportstech.github.io./XML_test_2.xml",initialLoadData)


function updateStatVis(stat,contentData,status){

	stat_id="stat_"+String(stat);

	if (status=="visible"){
		document.getElementById(stat_id).innerHTML="<strong>"+stat+": </strong>"+contentData.textContent;
		document.getElementById(stat_id).classList.add('statLine');
	}

	if (status=="hidden"){
		document.getElementById(stat_id).classList.add('hidden');
		document.getElementById(stat_id).classList.remove('statLine');
	}
}

function addStatSeperator(){
	let statSpans=document.getElementsByClassName('statLine');

	for (let i=0;i<statSpans.length-1;i++){
		let current=statSpans[i].innerHTML;
		statSpans[i].innerHTML=current+',';
	}

}



function loadData(input,buildID){

	
	var skillCounter=0;
	var playerNode= xmlDoc.evaluate("//Player[playerID = "+input+"]",xmlDoc,null,XPathResult.FIRST_ORDERED_NODE_TYPE,null);
	var actualNode=playerNode.singleNodeValue;
	
	if (actualNode.hasChildNodes()){
		var playerData=actualNode.children;


		for (let bld_id=1;bld_id<=3;bld_id++){
			let temp=document.getElementById('Build'+String(bld_id));
			
			var new_elem=temp.cloneNode(true);
			temp.parentNode.replaceChild(new_elem,temp)

			new_elem.addEventListener("click",function(){
				loadData(input,bld_id);
			})

		}


		for (let i=0;i<playerData.length;i++){
			
			if (playerData[i].tagName=="Name"){
				document.getElementById("playerName").innerHTML =playerData[i].textContent;
				
			};
			
			if (playerData[i].tagName=="Class"){
				
				document.getElementById("className").innerHTML =playerData[i].textContent;
				
			};
			
			
			if (playerData[i].tagName=="Class_Img"){
				document.getElementById("classIcon").src=playerData[i].textContent;
				
			};
			
			
			if (playerData[i].tagName.includes("Build")&& playerData[i].tagName.slice(-1)==buildID){//pull skill data only if build id matches
				//once inside, cycle through children
				let buildData=playerData[i].children;
				
				for (let b_id=0;b_id<buildData.length;b_id++){
					
					if (buildData[b_id].tagName=="Stats"){
						let statData=buildData[b_id].children;
						
						for (let s_id=0;s_id<statData.length;s_id++){

							if (statData[s_id].tagName=="Agility" && statData[s_id].textContent!=''){
								updateStatVis("Agility",statData[s_id],"visible")
								
							}else if (statData[s_id].tagName=="Agility" && statData[s_id].textContent==''){
								updateStatVis("Agility",statData[s_id],"hidden")
							}
							
							if (statData[s_id].tagName=="Domination" && statData[s_id].textContent!=''){
								updateStatVis("Domination",statData[s_id],"visible")
								
							}else if (statData[s_id].tagName=="Domination" && statData[s_id].textContent==''){
								updateStatVis("Domination",statData[s_id],"hidden")
							}
							
							if (statData[s_id].tagName=="Crit" && statData[s_id].textContent!=''){
								updateStatVis("Crit",statData[s_id],"visible")
							}else if (statData[s_id].tagName=="Crit" && statData[s_id].textContent==''){
								updateStatVis("Crit",statData[s_id],"hidden")
							}
							
							if (statData[s_id].tagName=="Specialization" && statData[s_id].textContent!=''){
								updateStatVis("Specialization",statData[s_id],"visible")
							}else if (statData[s_id].tagName=="Specialization" && statData[s_id].textContent==''){
								updateStatVis("Specialization",statData[s_id],"hidden")
							}
							
							if (statData[s_id].tagName=="Expertise" && statData[s_id].textContent!=''){
								updateStatVis("Expertise",statData[s_id],"visible")
								
							}else if (statData[s_id].tagName=="Expertise" && statData[s_id].textContent==''){
								updateStatVis("Expertise",statData[s_id],"hidden")
							}
							
							if (statData[s_id].tagName=="Endurance" && statData[s_id].textContent!=''){
								updateStatVis("Endurance",statData[s_id],"visible")
								
							}else if (statData[s_id].tagName=="Endurance" && statData[s_id].textContent==''){
								updateStatVis("Endurance",statData[s_id],"hidden")
							}
						}

						addStatSeperator();
					}

					if (buildData[b_id].tagName.includes("Skill")){
						
						let skillNum=buildData[b_id].tagName.slice(-1);
						
						skillCounter=skillCounter+1;
						
						/*create new elements only when needed*/
						
						if (document.getElementById("skillList_"+String(skillNum))==null){
							const newElem=document.createElement('table');
			
							let temp_id=String(skillNum);
							
							
							newElem.className='skillListing';
							newElem.id='skillList_'+temp_id;
							
							
							newElem.addEventListener("click",function(){
								
								 $('#'+newElem.id+'>tbody>tr').siblings('.child-row'+temp_id+'0').toggle()
								
							})
							
							newElem.title="Click to expand/collapse";
							
				
							newElem.innerHTML=`<table><tr class='parent' id='row`+temp_id+`0'><td colspan='4'><table><tr><td class='skill' colspan='1'><div style='text-align:center'><img class='skillIcon' id='skillIcon_`+temp_id+`' />
												</div></td><td colspan='3'><span class='skillTitle' id='skillName_`+temp_id+`'></span><br><span class='skillTitle' id='skillPoints_`+temp_id+`'></span></td></tr><tr><td colspan='4'><div class='container'>
												<span><img class='tripIcon' id='tripIcon_`+temp_id+`_1' /></span><span><img class='tripIcon' id='tripIcon_`+temp_id+`_2'/></span><span><img class='tripIcon' id='tripIcon_`+temp_id+`_3' /></span></div></td>
												</tr></table></td></tr><tr class='child-row`+temp_id+`0' style='display: none;'><td colspan='4'><div class='descriptor'><strong>Description:</strong></div><div class='descriptor' id='skillDescrip_`+temp_id+`'></div></td></tr>
												<tr class='child-row`+temp_id+`0' style='display: none;'><td colspan='4'><div class='descriptor'><img class='tripIcon' id='tripIconDetail_`+temp_id+`_1' />
												<span class='descriptor' id='tripName_`+temp_id+`_1'></span><div class='descriptor' id='tripDescrip_`+temp_id+`_1'></div></div></td></tr><tr class='child-row`+temp_id+`0' style='display: none;'><td colspan='4'><div class='descriptor'><img class='tripIcon' id='tripIconDetail_`+temp_id+`_2' />
												<span class='descriptor' id='tripName_`+temp_id+`_2'></span><div class='descriptor' id='tripDescrip_`+temp_id+`_2'></div></div></td></tr><tr class='child-row`+temp_id+`0' style='display: none;'><td colspan='4'><div class='descriptor'><img class='tripIcon' id='tripIconDetail_`+temp_id+`_3' />
												<span class='descriptor' id='tripName_`+temp_id+`_3'></span><div class='descriptor' id='tripDescrip_`+temp_id+`_3'></div></div></td></tr></table>`
							
							document.getElementById('skillTable1').appendChild(newElem)

						} else {
							/* console.log('Element '+String(skillNum)+' already here') */
							document.getElementById("skillList_"+String(skillNum)).style.display="table";

						}

						let skillData=buildData[b_id].children;
						
						for (let j=0;j<skillData.length;j++){
							
							if (skillData[j].tagName=="skillName"){
								
								document.getElementById("skillName_"+String(skillNum)).innerHTML=skillData[j].textContent;
							};
							
							/*removing points for now*/
							/* if (skillData[j].tagName=="points"){
								document.getElementById("skillPoints_0"+String(pID)+'_'+String(skillNum)).innerHTML=skillData[j].textContent+String("pts");
							}; */
							
							if (skillData[j].tagName=="skillDescrip"){
								document.getElementById("skillDescrip_"+String(skillNum)).innerHTML=skillData[j].textContent;
							};
							
							if (skillData[j].tagName=="skillIcon"){
								document.getElementById("skillIcon_"+String(skillNum)).src=skillData[j].textContent;
							};
							
							if (skillData[j].tagName.includes("Tripod")){
								let tripNum=skillData[j].tagName.slice(-1)
								let tripData=skillData[j].children;

								
								for (let k=0;k<tripData.length;k++){
									
									if (tripData[k].tagName=="tripodName"){
										document.getElementById("tripName_"+String(skillNum)+'_'+String(tripNum)).innerHTML="<Strong>"+tripData[k].textContent+"</Strong>";
									};
									

									if (tripData[k].tagName=="tripodDescrip"){
										document.getElementById("tripDescrip_"+String(skillNum)+'_'+String(tripNum)).innerHTML=tripData[k].textContent;
									};
									
									if (tripData[k].tagName=="tripodIcon"){
										
										if (tripData[k].textContent!=""){
											document.getElementById("tripIcon_"+String(skillNum)+'_'+String(tripNum)).hidden=false;
											document.getElementById("tripIcon_"+String(skillNum)+'_'+String(tripNum)).src=tripData[k].textContent;
											
											document.getElementById("tripIconDetail_"+String(skillNum)+'_'+String(tripNum)).hidden=false;
											document.getElementById("tripIconDetail_"+String(skillNum)+'_'+String(tripNum)).src=tripData[k].textContent;
										} else{
											document.getElementById("tripIcon_"+String(skillNum)+'_'+String(tripNum)).hidden=true;
											document.getElementById("tripIconDetail_"+String(skillNum)+'_'+String(tripNum)).hidden=true;
										};
									};
								};
							};
						};
					};
				};
			};
		
		};
		
		let totalSkillTables=document.getElementsByClassName('skillListing');
		let totalSkillNum=totalSkillTables.length;
		
		while (skillCounter<totalSkillNum){
			let sk_id=skillCounter+1;
			document.getElementById("skillList_"+String(sk_id)).style.display="none";
			
			skillCounter=skillCounter+1;
		}
		
		
	};
	
	
}

/*add player button functionality, defaults to build 1 on player load*/
for (let pb_id=01;pb_id<=10;pb_id++){
	let temp=document.getElementById('Player'+String(pb_id));
	
	temp.addEventListener("click",function(){
		loadData(pb_id,1);
	})

}




dragElement(document.getElementById("dragdiv1"));
/* dragElement(document.getElementById("dragdiv2")); */
function dragElement(elmnt) {
  var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  
  if (document.getElementById(elmnt.id + "header")) {
    /* if present, the header is where you move the DIV from:*/
    document.getElementById(elmnt.id + "header").onmousedown = dragMouseDown;
  } else {
    /* otherwise, move the DIV from anywhere inside the DIV:*/
    elmnt.onmousedown = dragMouseDown;
  }
  
  //check if element is visible
  function correctVis(elmnt){
	
	var bounding=elmnt.getBoundingClientRect();
	
	return (bounding.top >= 0 &&
        bounding.left >= 0 &&
        bounding.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        bounding.right <= (window.innerWidth || document.documentElement.clientWidth)
		);
  };
  
  
  window.addEventListener('resize',function(){
	var bounding=elmnt.getBoundingClientRect();
	let topLimit=0;
	let leftLimit=0;
	let rightLimit=(window.innerWidth || document.documentElement.clientWidth);
	let botLimit=(window.innerHeight || document.documentElement.clientHeight);
	
	let origTop=bounding.top;
	let origLeft=bounding.left;
	  	
	if (bounding.top<topLimit){
		elmnt.style.top=0;
	}
	if (bounding.left<leftLimit){
		elmnt.style.left=0;
	}
	
	if (bounding.bottom>botLimit){
		elmnt.style.top=origTop-(bounding.bottom-botLimit) + "px"
	}
	
	if (bounding.right>rightLimit){
		elmnt.style.left=origLeft-(bounding.right-rightLimit) + "px"
		/* console.log(elmnt.style.left) */
	}
		
  })

  function dragMouseDown(e) {
    e = e || window.event;
    e.preventDefault();
    // get the mouse cursor position at startup:
    pos3 = e.clientX;
    pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    // call a function whenever the cursor moves:
    document.onmousemove = elementDrag;
  }
  
//drag element, with limits based on current view window
  function elementDrag(e) {
	let topLimit=0;
	let leftLimit=0;
	let rightLimit=(window.innerWidth || document.documentElement.clientWidth);
	let botLimit=(window.innerHeight || document.documentElement.clientHeight);
	var bounding=elmnt.getBoundingClientRect();
	var newTop; var newLeft;
	
    e = e || window.event;
    e.preventDefault();
    // calculate the new cursor position:
    pos1 = pos3 - e.clientX;
    pos2 = pos4 - e.clientY;
    pos3 = e.clientX;
    pos4 = e.clientY;
    // set the element's new position:
	newTop=(elmnt.offsetTop - pos2);
	newLeft=(elmnt.offsetLeft - pos1);
	
	if (newTop<topLimit){
		newTop=topLimit;
	};
	
	if (newLeft<leftLimit){
		newLeft=leftLimit;
	};
	
	if (bounding.right>rightLimit){
		newLeft=(elmnt.offsetLeft - pos1)-(bounding.right-rightLimit);
	}
	
	if (bounding.bottom>botLimit){
		newTop=(elmnt.offsetTop-pos2)-(bounding.bottom-botLimit);
	}
	elmnt.style.top = newTop + "px";
    elmnt.style.left = newLeft + "px";
	
  }

  function closeDragElement() {
    /* stop moving when mouse button is released:*/
    document.onmouseup = null;
    document.onmousemove = null;
  }
}