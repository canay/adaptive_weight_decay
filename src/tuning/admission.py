"""Hash-bound precompute evidence: no hand-written PASS alone can launch main."""
from core import *

EVIDENCE_NAMES = ['PROSPECTIVE_FIELD_PREFLIGHT.json','DURABILITY_PLAN_PREFLIGHT.json',
                  'INPUT_QA.json','TEST_REPORT.json','SMOKE_GATE.json','REVIEW_RECEIPT.json']

def require(condition,message):
    if not condition:raise RuntimeError(message)

def evidence(root,cfg,frozen,environment):
    root=Path(root);candidate=load(root/'PROSPECTIVE_CANDIDATE.json');ch=sha(root/'PROSPECTIVE_CANDIDATE.json')
    require(candidate['source_bundle_sha256']==frozen,'CANDIDATE_SOURCE_DRIFT')
    records={name:load(root/name) for name in EVIDENCE_NAMES}
    require(all(r.get('status')=='PASS' for r in records.values()),'PREFLIGHT_NOT_PASS')
    require(records['PROSPECTIVE_FIELD_PREFLIGHT.json']['candidate_sha256']==ch,'PROSPECTIVE_DRIFT')
    require(not records['PROSPECTIVE_FIELD_PREFLIGHT.json']['errors'],'PROSPECTIVE_ERRORS')
    plan=records['DURABILITY_PLAN_PREFLIGHT.json']
    require(plan['durability_sha256']==sha(root/'DURABILITY.md') and plan['exit_code']==0,'DURABILITY_DRIFT')
    for name in ['INPUT_QA.json','TEST_REPORT.json','SMOKE_GATE.json']:
        r=records[name]
        require(r['freeze_sha256']==frozen and r['environment']==environment,'REMOTE_EVIDENCE_DRIFT '+name)
    require(len(records['INPUT_QA.json']['rows'])==18,'INPUT_CONTEXTS_INCOMPLETE')
    tests=records['TEST_REPORT.json']
    require(tests['tests_run']>=27 and tests['failures']==0 and tests['errors']==0,'TESTS_INCOMPLETE')
    smoke=records['SMOKE_GATE.json']
    require(bool(smoke['checks']) and all(v is True for v in smoke['checks'].values()),'SMOKE_CHECK_FAILED')
    require(0<smoke['conservative_eta_seconds']<cfg['watchdog_seconds'],'ETA_NOT_ADMITTED')
    for rel,h in smoke['evidence_files'].items():
        require(sha(root/rel)==h,'SMOKE_EVIDENCE_DRIFT '+rel)
    review=records['REVIEW_RECEIPT.json']
    require(review['freeze_sha256']==frozen and review['candidate_sha256']==ch,'REVIEW_SOURCE_DRIFT')
    require(review['unresolved_blocking_findings']==0 and review['permission_denials']==[], 'REVIEW_UNRESOLVED')
    require(review['fresh_session'] is True and review['no_session_persistence'] is True,'REVIEW_SESSION_CONTRACT')
    require(review['guard_verify_output_pass'] is True,'REVIEW_GUARD_NOT_PASS')
    for rel,h in review['evidence_files'].items():
        require(sha(root/rel)==h,'REVIEW_ARTIFACT_DRIFT '+rel)
    require(len(review['evidence_files'])>=3,'REVIEW_ARTIFACTS_INCOMPLETE')
    return {name:sha(root/name) for name in EVIDENCE_NAMES},ch

def verify_admission(root,cfg,frozen,environment):
    hashes,ch=evidence(root,cfg,frozen,environment);gate=load(Path(root)/'ADMISSION.json')
    require(gate.get('status')=='PASS' and gate.get('freeze_sha256')==frozen and
            gate.get('environment')==environment and gate.get('prospective_candidate_sha256')==ch and
            gate.get('evidence_files')==hashes,'MAIN_ADMISSION_NOT_CURRENT')
    return gate

def main():
    cfg=load(ROOT/'config.json');frozen=verify_freeze();environment=env()
    hashes,ch=evidence(ROOT,cfg,frozen,environment)
    body=dict(status='PASS',generated_utc=now(),run_id=cfg['run_id'],freeze_sha256=frozen,
              environment=environment,prospective_candidate_sha256=ch,evidence_files=hashes)
    exclusive(ROOT/'ADMISSION.json',body);verify_admission(ROOT,cfg,frozen,environment)
    print('MAIN_ADMISSION_PASS',sha(ROOT/'ADMISSION.json'))

if __name__=='__main__':main()
