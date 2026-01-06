{{/*
Expand the name of the chart.
*/}}
{{- define "islands.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "islands.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "islands.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "islands.labels" -}}
helm.sh/chart: {{ include "islands.chart" . }}
{{ include "islands.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "islands.selectorLabels" -}}
app.kubernetes.io/name: {{ include "islands.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "islands.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "islands.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "islands.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion }}
{{- printf "%s:%s" .Values.image.repository $tag }}
{{- end }}

{{/*
Return the PVC claim name
*/}}
{{- define "islands.pvcName" -}}
{{- if .Values.persistence.existingClaim }}
{{- .Values.persistence.existingClaim }}
{{- else }}
{{- include "islands.fullname" . }}-data
{{- end }}
{{- end }}

{{/*
Return the secret name
*/}}
{{- define "islands.secretName" -}}
{{- include "islands.fullname" . }}-secrets
{{- end }}

{{/*
Return the configmap name
*/}}
{{- define "islands.configmapName" -}}
{{- include "islands.fullname" . }}-config
{{- end }}
