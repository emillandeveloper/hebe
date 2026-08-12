from __future__ import annotations
import json,time,uuid
from typing import Any,Callable
from app.continuity.models import OpenThread,OpenThreadStatus
from app.epistemics.models import BeliefStatus,EvidenceRef,EvidenceRelation
from app.social_world_v2.models import CultureStatus,SocialContext,SocialEpisode,SocialOpportunity

class SocialWorldService:
    LOW_VALUE={"hola","hello","lol","xd","x d","que boss","qué boss"};HIGH_SENSITIVITY={"medical_diagnosis","sexual","political","religion","financial","legal","exact_location","private_relationship"}
    def __init__(self,repository,beliefs,threads,retrieval=None,*,now_fn:Callable[[],float]=time.time):self.repository=repository;self.beliefs=beliefs;self.threads=threads;self.retrieval=retrieval;self.now_fn=now_fn;self.last_resolution={};self.last_context={};self.last_opportunities=[];self.last_culture_selection={"selected":[],"rejected":[]};self.rejections=[]
    def resolve_person(self,*,platform="twitch",platform_user_id="",login="",display_name="",source="twitch",stream_session_id=""):
        person,identity,decision=self.repository.resolve_person(platform=platform,platform_user_id=platform_user_id,login=login,display_name=display_name,now=self.now_fn(),source=source)
        if stream_session_id:self.repository.record_session(person.person_id,stream_session_id,self.now_fn())
        self.last_resolution={"person":person.to_dict(),"identity":identity.to_dict(),"decision":decision};print(f"[HEBE][SOCIAL_PERSON_RESOLVE] platform={platform} platform_user_id={platform_user_id} person_id={person.person_id} decision={decision}",flush=True);return person,identity
    def record_episode(self,*,episode_type,participant_ids,origin_event_id,summary,salience_reason="",relevance_seconds=86400,retention_seconds=2592000,sensitivity="normal",retention_class="bounded",retrieval_scope="stream_public",related_event_ids=(),tone_observations=()):
        normalized=" ".join(str(summary or "").casefold().split())
        if not salience_reason or normalized in self.LOW_VALUE or sensitivity in self.HIGH_SENSITIVITY:
            reason="sensitive_nonpersistent" if sensitivity in self.HIGH_SENSITIVITY else "low_salience";self.rejections.append({"origin_event_id":origin_event_id,"reason":reason,"episode_type":episode_type});print(f"[HEBE][SOCIAL_EPISODE] origin={origin_event_id} admitted=false reason={reason}",flush=True);return None
        now=self.now_fn();e=SocialEpisode(f"social_episode_{uuid.uuid4().hex}",episode_type,tuple(participant_ids),origin_event_id,tuple(related_event_ids),str(summary)[:240],tuple(tone_observations),now,now+max(0,relevance_seconds),now+max(0,retention_seconds),sensitivity,retention_class,retrieval_scope,salience_reason)
        self.repository.save_episode(e);print(f"[HEBE][SOCIAL_EPISODE] episode_id={e.id} type={episode_type} participants={list(participant_ids)} salience={salience_reason} retention={retention_class}",flush=True);return e
    def propose_hypothesis(self,person_id,*,predicate,object_value,confidence,evidence,sensitivity="normal",relevance_seconds=2592000):
        if sensitivity in self.HIGH_SENSITIVITY:self.rejections.append({"source_event_id":evidence.source_event_id,"reason":"sensitive_hypothesis_rejected"});return None
        b=self.beliefs.propose(namespace="social",scope_kind="person",scope_id=person_id,subject_ref=person_id,predicate=predicate,object_value=object_value,confidence=confidence,authority_class="extractor",evidence=evidence,status=BeliefStatus.INFERRED,sensitivity=sensitivity,relevance_until=self.now_fn()+relevance_seconds)
        if b:print(f"[HEBE][SOCIAL_HYPOTHESIS] belief_id={b.id} predicate={predicate} status={b.epistemic_status.value} confidence={b.confidence:.3f} evidence={list(b.evidence_ids)}",flush=True)
        return b
    def correct_hypothesis(self,belief_id,*,object_value,evidence):return self.beliefs.correct(belief_id,object_value=object_value,evidence=evidence,authority_class="owner")
    def open_social_thread(self,person_id,*,thread_type,subject_ref,summary,origin_event_id,relevance_seconds,valid_seconds=None,sensitivity="normal",priority=40):
        now=self.now_fn();thread=OpenThread(f"thread_{uuid.uuid4().hex}",thread_type,"person",person_id,(person_id,),subject_ref,summary,origin_event_id,origin_event_id,OpenThreadStatus.OPEN,priority,now,now+relevance_seconds,now+(valid_seconds if valid_seconds is not None else relevance_seconds),0,"",sensitivity,1);self.threads.create(thread);print(f"[HEBE][SOCIAL_THREAD_OPEN] thread_id={thread.id} person_id={person_id} type={thread_type} relevance_until={thread.relevance_until}",flush=True);return thread
    def resolve_social_thread(self,subject_ref,*,event_id,status=OpenThreadStatus.RESOLVED):return self.threads.transition_for_subject(subject_ref,status=status,event_id=event_id,now=self.now_fn())
    def _active_threads(self,**kwargs):
        started=time.perf_counter();result=self.threads.list_open(**kwargs);self.repository.latencies["thread_lookup"].append((time.perf_counter()-started)*1000);return result
    def expire_social_threads(self,event_id="social_maintenance"):
        now=self.now_fn();count=0
        for thread in self._active_threads():
            if thread.scope_kind=="person" and min(thread.relevance_until,thread.valid_until)<=now:
                self.threads.transition(thread.id,expected_version=thread.version,status=OpenThreadStatus.EXPIRED,event_id=event_id,now=now);count+=1;print(f"[HEBE][SOCIAL_THREAD_EXPIRE] thread_id={thread.id}",flush=True)
        return count
    def opportunities(self,person_id,*,scene_suitable=True):
        now=self.now_fn();items=[]
        for t in self._active_threads(scope_kind="person",scope_id=person_id,now=now):
            if t.relevance_until<=now:continue
            items.append(SocialOpportunity(f"opportunity_{t.id}","open_thread_followup",person_id,(t.origin_event_id,),.8,.9,"low",t.sensitivity,"question_followup",scene_suitable,False).to_dict())
        self.last_opportunities=items;return items
    def create_culture_candidate(self,*,label,meaning,participant_ids,origin_episode_id,event_id,tone="playful",scope="participants",owner_confirmed=False):
        now=self.now_fn();item={"id":f"culture_{uuid.uuid4().hex}","label":label,"meaning":meaning,"origin_episode_id":origin_episode_id,"participant_ids":list(participant_ids),"scope":scope,"tone":tone,"status":CultureStatus.ACTIVE.value if owner_confirmed else CultureStatus.CANDIDATE.value,"confidence":1.0 if owner_confirmed else .45,"created_at":now,"last_reinforced_at":now,"last_used_at":0.0,"reuse_count":0,"cooldown_until":0.0};self.repository.save_culture(item);self.repository.add_culture_evidence(item["id"],event_id,origin_episode_id,"origin","positive",1 if owner_confirmed else .5,now,"owner" if owner_confirmed else "interaction");print(f"[HEBE][CULTURE_CANDIDATE] item_id={item['id']} status={item['status']}",flush=True);return item
    def reinforce_culture(self,item_id,*,event_id,episode_id="",reaction="positive",weight=1.0,authority="interaction"):
        item=self.repository.culture(item_id);now=self.now_fn();polarity="negative" if reaction in {"negative","dislike","owner_reject"} else "positive";self.repository.add_culture_evidence(item_id,event_id,episode_id,reaction,polarity,weight,now,authority);evidence=self.repository.culture_evidence(item_id);positive=sum(float(x["weight"]) for x in evidence if x["polarity"]=="positive");negative=sum(float(x["weight"]) for x in evidence if x["polarity"]=="negative")
        if polarity=="negative":status=CultureStatus.RETIRED.value if authority=="owner" or negative>=1.5 else CultureStatus.WEAKENING.value;confidence=max(0,float(item["confidence"])-.35*weight)
        else:status=CultureStatus.ACTIVE.value if positive>=2.5 else item["status"];confidence=min(1,float(item["confidence"])+.18*weight)
        item.update(status=status,confidence=confidence,last_reinforced_at=now);self.repository.save_culture(item);print(f"[HEBE][CULTURE_REINFORCE] item_id={item_id} status={status} confidence={confidence:.3f}",flush=True);return item
    def use_culture(self,item_id,*,event_id,cooldown_seconds=3600):
        item=self.repository.culture(item_id);now=self.now_fn()
        if item["status"]!=CultureStatus.ACTIVE.value or item["cooldown_until"]>now:return None
        item.update(last_used_at=now,reuse_count=int(item["reuse_count"])+1,cooldown_until=now+cooldown_seconds);self.repository.save_culture(item);self.repository.add_culture_evidence(item_id,event_id,"","used","neutral",0,now,"system");print(f"[HEBE][CULTURE_USE] item_id={item_id} cooldown_until={item['cooldown_until']}",flush=True);return item
    def select_culture(self,person_id,*,topic="",scene_tone="casual"):
        start=time.perf_counter();now=self.now_fn();selected=[];rejected=[]
        for item in self.repository.culture():
            reason=""
            if person_id not in item["participant_ids"]:reason="participant_mismatch"
            elif item["status"]!=CultureStatus.ACTIVE.value:reason="not_active"
            elif item["cooldown_until"]>now:reason="cooldown"
            elif not topic or not any(token in topic.casefold() for token in (item["label"].casefold(),item["meaning"].casefold())):reason="context_mismatch"
            elif scene_tone in {"serious","sensitive"}:reason="tone_mismatch"
            (rejected if reason else selected).append({**item,**({"rejection_reason":reason} if reason else {})})
        self.repository.latencies["culture_select"].append((time.perf_counter()-start)*1000);self.last_culture_selection={"selected":selected,"rejected":rejected};return selected,rejected
    def retrieve_social_context(self,person_id,*,purpose="social_greeting",retrieval_scope="stream_public",topic="",scene_tone="casual"):
        start=time.perf_counter();now=self.now_fn();selected=[];rejected=[];reasons={};episodes=[]
        for e in self.repository.episodes(person_id):
            reason=""
            if e["retention_until"] and e["retention_until"]<=now:reason="retention_expired"
            elif e["relevance_until"]<=now:reason="relevance_expired"
            elif retrieval_scope=="stream_public" and (e["retrieval_scope"]!="stream_public" or e["sensitivity"] not in {"normal","low"}):reason="privacy_scope"
            if reason:e["rejection_reason"]=reason;rejected.append(e);reasons[reason]=reasons.get(reason,0)+1
            else:episodes.append(e);selected.append(e)
        beliefs=[]
        for b in self.beliefs.repository.list(namespace="social",scope_kind="person",scope_id=person_id):
            item=b.to_dict();reason=""
            if b.relevance_until and b.relevance_until<=now:reason="relevance_expired"
            elif retrieval_scope=="stream_public" and b.sensitivity!="normal":reason="privacy_scope"
            if reason:item["rejection_reason"]=reason;rejected.append(item);reasons[reason]=reasons.get(reason,0)+1
            elif b.epistemic_status not in {BeliefStatus.SUPERSEDED,BeliefStatus.REJECTED,BeliefStatus.HISTORICAL}:beliefs.append(item);selected.append(item)
        threads=[t.to_dict() for t in self._active_threads(scope_kind="person",scope_id=person_id,now=now) if t.relevance_until>now]
        culture,culture_rejected=self.select_culture(person_id,topic=topic,scene_tone=scene_tone);rejected.extend(culture_rejected)
        for x in culture_rejected:reasons[x["rejection_reason"]]=reasons.get(x["rejection_reason"],0)+1
        selected.extend(culture);manifest=[{"id":x.get("id"),"source_class":"episode" if str(x.get("id","")).startswith("social_episode") else "social_belief" if str(x.get("id","")).startswith("belief") else "shared_culture","status":x.get("epistemic_status") or x.get("status"),"confidence":x.get("confidence"),"sensitivity":x.get("sensitivity","normal"),"evidence_ids":x.get("evidence_ids") or [x.get("origin_event_id")]} for x in selected]
        latency=(time.perf_counter()-start)*1000;self.repository.latencies["context"].append(latency);context=SocialContext(self.repository.person(person_id),self.repository.familiarity(person_id),tuple(episodes[:5]),tuple(threads[:5]),tuple(beliefs[:5]),tuple(culture[:3]),(),tuple(selected[:10]),tuple(rejected[:20]),reasons,tuple(manifest),len(json.dumps(manifest,ensure_ascii=False).encode()),latency);self.last_context=context.to_dict();print(f"[HEBE][SOCIAL_RETRIEVE] purpose={purpose} selected={len(selected)} rejected={reasons}",flush=True);return context
